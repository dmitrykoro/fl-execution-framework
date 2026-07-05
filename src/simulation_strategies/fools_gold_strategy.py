from __future__ import annotations

import os
import time
import logging

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl

from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from src.output_handlers.directory_handler import DirectoryHandler
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics


@dataclass
class FoolsGoldConfig:
    """FoolsGold hyperparameters and integration switches."""
    # If True: weight by alpha_i * num_examples. If False: weight by alpha_i only.
    weight_by_num_examples: bool = True

    # Numerical stability
    eps: float = 1e-12

    # Optional: keep a running sum of update vectors per-client (classic FoolsGold uses history)
    use_update_history: bool = True

    # Optional: cap or floor alpha values
    alpha_min: float = 0.0
    alpha_max: float = 1.0


class FoolsGoldStrategy(fl.server.strategy.FedAvg):
    """
    FoolsGold robust aggregation strategy.
    - Computes similarity between client update vectors (or their history).
    - Downweights clients whose updates are consistently too similar (Sybil/collusion signal).

    References:
      Fung et al., "Mitigating Sybils in Federated Learning Poisoning" (FoolsGold)

    This strategy is implemented as a server-side robust aggregator.
    It does NOT remove clients by default; it reweights them.

    Compatible with Flower simulations; intended as a drop-in Strategy replacement.
    """

    def __init__(
            self,
            *,
            fg: Optional[FoolsGoldConfig] = None,
            strategy_history: Optional[SimulationStrategyHistory] = None,
            remove_clients: bool = False,  # kept for compatibility with your framework
            begin_removing_from_round: Optional[int] = None,  # kept for compatibility
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.fg = fg or FoolsGoldConfig()
        self.strategy_history = strategy_history

        # Compatibility flags (FoolsGold usually doesn't remove, but your framework expects these)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round

        self.current_round: int = 0
        self.removed_client_ids: set[str] = set()

        # We need the last global parameters to compute client updates
        self._last_global: Optional[List[np.ndarray]] = None

        # Update history per client (sum of past update vectors)
        self._client_update_history: Dict[str, np.ndarray] = {}

        # Logger (same style as your PID strategy)
        self.logger = logging.getLogger(f"foolsgold_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)

        out_dir = DirectoryHandler.dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()

        # Avoid double-handlers if created multiple times in Ray workers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        self.logger.propagate = False

    # -----------------------------
    # Helpers
    # -----------------------------

    @staticmethod
    def _flatten_params(params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.reshape(-1) for p in params]).astype(np.float64, copy=False)

    @staticmethod
    def _cosine_sim_matrix(X: np.ndarray, eps: float) -> np.ndarray:
        """
        X: shape (n_clients, d)
        returns cosine similarity matrix shape (n_clients, n_clients)
        """
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
        Xn = X / norms
        return Xn @ Xn.T

    def _foolsgold_alphas(self, vectors: np.ndarray) -> np.ndarray:
        """
        Core FoolsGold weighting:
          - compute cosine similarities
          - pardoning
          - alpha_i = 1 - max_sim_i
          - rescale + logit shaping
        """
        n = vectors.shape[0]
        if n == 1:
            return np.array([1.0], dtype=np.float64)

        cs = self._cosine_sim_matrix(vectors, self.fg.eps)

        # Zero out self-similarity
        np.fill_diagonal(cs, 0.0)

        maxcs = np.max(cs, axis=1) + self.fg.eps

        # Pardoning
        # If client j is "more suspicious" (higher max similarity) than i, reduce i's similarity to j.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i, j] *= maxcs[i] / maxcs[j]

        # Compute alpha: 1 - max similarity after pardoning
        maxcs2 = np.max(cs, axis=1)
        alpha = 1.0 - maxcs2

        # Clip to [0, 1]
        alpha = np.clip(alpha, self.fg.alpha_min, self.fg.alpha_max)

        # Normalize by max (avoid all zeros)
        amax = float(np.max(alpha))
        if amax > self.fg.eps:
            alpha = alpha / amax
        alpha = np.clip(alpha, self.fg.alpha_min, self.fg.alpha_max)

        # Logit transform (as in common FoolsGold implementations)
        # alpha = log(alpha/(1-alpha)) + 0.5 then clip to [0,1]
        # guard against 0 or 1
        alpha = np.clip(alpha, self.fg.eps, 1.0 - self.fg.eps)
        alpha = np.log(alpha / (1.0 - alpha)) + 0.5
        alpha = np.clip(alpha, self.fg.alpha_min, self.fg.alpha_max)

        return alpha.astype(np.float64, copy=False)

    def _aggregate_weighted_ndarrays(
            self,
            client_params: List[List[np.ndarray]],
            weights: np.ndarray,
    ) -> List[np.ndarray]:
        """Weighted average over ndarrays (same shapes), weights shape (n_clients,)."""
        wsum = float(np.sum(weights))
        if wsum <= self.fg.eps:
            # Fallback: uniform average
            weights = np.ones_like(weights, dtype=np.float64)
            wsum = float(np.sum(weights))

        num_layers = len(client_params[0])
        out: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            stacked = np.stack([cp[layer_idx] for cp in client_params], axis=0)  # (n, ...)
            # Weighted sum
            layer = np.tensordot(weights, stacked, axes=(0, 0)) / wsum
            out.append(layer.astype(stacked.dtype, copy=False))

        return out

    # -----------------------------
    # Flower Strategy Overrides
    # -----------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        # Store last global parameters so we can compute updates in aggregate_fit
        self._last_global = fl.common.parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        if self._last_global is None:
            # Fallback if configure_fit was bypassed
            # Use FedAvg aggregation (or infer last_global from first client)
            return super().aggregate_fit(server_round, results, failures)

        # Filter removed clients (kept for compatibility)
        aggregate_clients: List[Tuple[ClientProxy, FitRes]] = [
            r for r in results if r[0].cid not in self.removed_client_ids
        ]
        if not aggregate_clients:
            return super().aggregate_fit(server_round, results, failures)

        # Build per-client update vectors
        client_ids: List[str] = []
        client_params: List[List[np.ndarray]] = []
        update_vectors: List[np.ndarray] = []
        num_examples: List[int] = []

        for client_proxy, fit_res in aggregate_clients:
            cid = client_proxy.cid
            client_ids.append(cid)
            num_examples.append(int(fit_res.num_examples))

            cp = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_params.append(cp)

            # Update = client_params - last_global (flattened)
            upd = self._flatten_params(cp) - self._flatten_params(self._last_global)

            if self.fg.use_update_history:
                prev = self._client_update_history.get(cid)
                if prev is None:
                    self._client_update_history[cid] = upd
                else:
                    self._client_update_history[cid] = prev + upd
                upd_used = self._client_update_history[cid]
            else:
                upd_used = upd

            update_vectors.append(upd_used)

        X = np.stack(update_vectors, axis=0)  # (n_clients, d)
        t0 = time.time_ns()
        alphas = self._foolsgold_alphas(X)
        t1 = time.time_ns()

        # Effective weights
        if self.fg.weight_by_num_examples:
            weights = alphas * np.array(num_examples, dtype=np.float64)
        else:
            weights = alphas

        # Aggregate using weights
        aggregated_ndarrays = self._aggregate_weighted_ndarrays(client_params, weights)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # Metrics for logging/history
        metrics_aggregated: Dict[str, Scalar] = {
            "aggregated_with": "foolsgold",
            "fg_weight_by_num_examples": bool(self.fg.weight_by_num_examples),
            "fg_use_update_history": bool(self.fg.use_update_history),
            "fg_score_calc_time_nanos": int(t1 - t0),
            "num_aggregated_clients": int(len(aggregate_clients)),
            "alpha_mean": float(np.mean(alphas)),
            "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas)),
        }

        # Write per-client history (if your history object supports it)
        if self.strategy_history:
            self.strategy_history.insert_round_history_entry(
                score_calculation_time_nanos=int(t1 - t0)
            )

            for cid, a in zip(client_ids, alphas):
                # Store alpha as the "removal_criterion" field (it is a suspiciousness/weight score)
                # If you prefer, change field name in your DB model.
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    removal_criterion=float(1.0 - a),  # higher means more suspicious
                )

            # Also store a round-level “threshold” placeholder (optional)
            self.strategy_history.insert_round_history_entry(removal_threshold=float(np.mean(alphas)))

        # Log
        self.logger.info("\n" + "-" * 40 + f" FOOLSGOLD ROUND {server_round} " + "-" * 40)
        for cid, a, w in zip(client_ids, alphas, weights):
            self.logger.info(f"Client {cid} alpha={a:.6f} weight={w:.3f}")
        self.logger.info(f"FoolsGold: alpha_mean={metrics_aggregated['alpha_mean']:.6f}")

        # Update last global to the new global for next round update computation
        self._last_global = aggregated_ndarrays

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        # Mirror your PID strategy behavior: record per-client metrics, then aggregate
        self.logger.info("\n" + "-" * 40 + f" EVAL ROUND {server_round} " + "-" * 40)

        if self.strategy_history:
            for client_proxy, evaluate_res in results:
                cid = client_proxy.cid
                if evaluate_res.metrics and "accuracy" in evaluate_res.metrics:
                    self.strategy_history.insert_single_client_history_entry(
                        client_id=int(cid),
                        current_round=self.current_round,
                        accuracy=float(evaluate_res.metrics["accuracy"]),
                    )
                self.strategy_history.insert_single_client_history_entry(
                    client_id=int(cid),
                    current_round=self.current_round,
                    loss=float(evaluate_res.loss),
                )

        if not results:
            return None, {}

        # Aggregate loss over non-removed clients (compat)
        losses = []
        n_loss_clients = 0
        for client_proxy, evaluate_res in results:
            if client_proxy.cid not in self.removed_client_ids:
                losses.append((evaluate_res.num_examples, evaluate_res.loss))
                n_loss_clients += 1

        loss_aggregated = weighted_loss_avg(losses) if losses else None
        if self.strategy_history and loss_aggregated is not None:
            self.strategy_history.insert_round_history_entry(loss_aggregated=float(loss_aggregated))

        # Aggregate metrics using your helper (keeps consistency with existing tooling)
        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=self.removed_client_ids,
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.begin_removing_from_round,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f"Round: {server_round} "
            f"Number of aggregated eval clients: {n_loss_clients} "
            f"Aggregated loss: {loss_aggregated}"
        )

        return loss_aggregated, metrics_aggregated
