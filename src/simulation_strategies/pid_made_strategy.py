"""
PID-MADE: PID with Model-Adaptive Deviation Estimation.

Spec: pid_made_spec.md

Key differences from existing PID strategies
---------------------------------------------
1. **Lagged centroid reference**
   - Warmup rounds (t <= T_warmup): reference = geometric median of current
     client weight vectors (robust against Byzantine outliers at startup).
   - Post-warmup: reference = the *previous* global model w (cheap, no extra
     communication, naturally adapts as the global model evolves).

2. **Discounted integral**
   - Each round every client's integral is decayed by λ before adding the new
     deviation: I[i] ← I[i]*λ + Δᵢ.
   - λ ∈ (0,1) controls how fast old history is forgotten (contrast with the
     existing strategies which accumulate raw distance sums indefinitely).

3. **Adaptive threshold** (three modes)
   - fixed    : constant τ supplied in the config.
   - chebyshev: distribution-free bound τ = μ + √((1-α)/α)·σ.
   - gaussian : parametric bound τ = μ + Φ⁻¹(1-α)·σ.

4. **Stateless per-round filtering**
   - Q = {i : Sᵢ ≤ τₜ}; if Q is empty, fall back to all participants.
   - No client is *permanently* removed — every client is re-evaluated fresh
     each round from its current score.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from scipy.stats import norm as scipy_norm

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics


class PIDMADEStrategy(fl.server.strategy.FedAvg):
    """
    PID-MADE aggregation strategy.

    Parameters
    ----------
    remove_clients : bool
        Whether to exclude high-score clients from aggregation.
    begin_removing_from_round : int
        Number of warmup rounds (T_warmup). During warmup the reference
        centroid is the geometric median; afterwards it is the lagged global
        model w.
    kp, ki, kd : float
        PID gain coefficients (proportional, integral, derivative).
    lam : float
        Forgetting factor λ ∈ (0, 1) applied to the per-client integral each
        round: I[i] ← I[i]*λ + Δᵢ.
    threshold_mode : str
        One of 'fixed', 'chebyshev', 'gaussian' — controls how τₜ is derived
        from the current round's score distribution.
    tau : float
        Constant threshold used when threshold_mode == 'fixed'.
    alpha : float
        Per-round false-positive rate α ∈ (0, 1) used by the chebyshev and
        gaussian threshold modes.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    network_model, use_lora, aggregation_strategy_keyword, output_dirname
        Standard framework plumbing (same across all strategies).
    """

    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        kp: float,
        ki: float,
        kd: float,
        lam: float,
        threshold_mode: str,
        tau: float,
        alpha: float,
        strategy_history: SimulationStrategyHistory,
        network_model,
        use_lora: bool,
        aggregation_strategy_keyword: str,
        output_dirname: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID-MADE specific
        self.lam = lam
        self.threshold_mode = threshold_mode
        self.tau = tau      # used only when threshold_mode == 'fixed'
        self.alpha = alpha  # used only when threshold_mode in {chebyshev, gaussian}

        # Framework params
        self.remove_clients = remove_clients
        self.t_warmup = begin_removing_from_round
        self.strategy_history = strategy_history
        self.network_model = network_model
        self.use_lora = use_lora
        self.aggregation_strategy_keyword = aggregation_strategy_keyword

        # Server state (spec: I, Delta_prev, ref, t)
        self.current_round: int = 0
        self.current_global_weights: Optional[List[np.ndarray]] = None  # w (lagged)
        self.client_integrals: Dict[str, float] = {}    # I[i]
        self.client_delta_prev: Dict[str, float] = {}   # Delta_prev[i]
        self.client_pid_scores: Dict[str, float] = {}   # Sᵢ from last round
        self.current_threshold: float = 0.0
        self.removed_client_ids: set = set()            # reset every round (stateless)

        # Logger (same pattern as other strategies)
        self.logger = logging.getLogger(f"pid_made_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)
        os.makedirs(output_dirname, exist_ok=True)
        self.logger.addHandler(logging.FileHandler(f"{output_dirname}/output.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # Sub-procedures (directly from spec)
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(weights: List[np.ndarray]) -> np.ndarray:
        """Concatenate all weight arrays into a single 1-D float64 vector."""
        return np.concatenate([w.flatten().astype(np.float64) for w in weights])

    @staticmethod
    def _geometric_median(
        vectors: List[np.ndarray],
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """
        Weiszfeld's algorithm for the geometric median.

        Spec:
            c ← MEAN(vectors)
            for _ in 1..max_iter:
                dists   ← max(||mat - c||₂ axis=1, tol)   # vectorized over all clients
                weights ← (1/dists) / sum(1/dists)
                c_new   ← weighted average of mat rows  # np.average, single C-level op
                if ||c_new - c||₂ < tol: break
                c ← c_new
            return c
        """
        mat = np.array(vectors)            # (n, d)
        c = np.mean(mat, axis=0)
        for _ in range(max_iter):
            dists = np.maximum(np.linalg.norm(mat - c, axis=1), tol)  # (n,) vectorized
            weights = (1.0 / dists) / np.sum(1.0 / dists)
            c_new = np.average(mat, axis=0, weights=weights)           # single C-level op
            if np.linalg.norm(c_new - c) < tol:
                break
            c = c_new
        return c

    def _compute_threshold(self, scores: List[float]) -> float:
        """
        Compute τₜ from the current round's score distribution.

        Spec:
            if mode == 'fixed':   return tau
            mu  ← MEAN(scores)
            std ← STD(scores)          # unbiased
            if mode == 'chebyshev': z ← sqrt((1-alpha)/alpha)
            if mode == 'gaussian':  z ← NORMAL_QUANTILE(1-alpha)
            return mu + z * std
        """
        if self.threshold_mode == "fixed":
            return self.tau

        mu = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

        if self.threshold_mode == "chebyshev":
            z = float(np.sqrt((1.0 - self.alpha) / self.alpha))
        else:  # gaussian
            z = float(scipy_norm.ppf(1.0 - self.alpha))

        return mu + z * std

    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Capture the current global weights (w) as the lagged reference for
        the upcoming round, then build the client list — excluding high-score
        clients when remove_clients is enabled (stateless: re-evaluated fresh
        each round).
        """
        # Capture w before clients train on it — this is the lagged reference
        # used in aggregate_fit step 2 after warmup.
        self.current_global_weights = fl.common.parameters_to_ndarrays(parameters)

        available_clients = client_manager.all()

        # Stateless: reset exclusion set every round
        self.removed_client_ids = set()

        if self.remove_clients and self.current_threshold > 0:
            for cid, score in self.client_pid_scores.items():
                if cid in available_clients and score > self.current_threshold:
                    self.removed_client_ids.add(cid)
                    self.logger.info(
                        f"Round {server_round}: excluding client {cid} "
                        f"(score {score:.4f} > τ {self.current_threshold:.4f})"
                    )

        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )
        self.logger.info(f"Round {server_round}: excluded = {self.removed_client_ids}")

        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(client, fit_ins) for client in available_clients.values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        time_start = time.time_ns()

        # ── Extract flattened weight vectors ──────────────────────────────
        client_ids: List[str] = []
        flat_weights: List[np.ndarray] = []
        weight_lists: List[List[np.ndarray]] = []

        for client_proxy, fit_res in results:
            w_list = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_ids.append(client_proxy.cid)
            weight_lists.append(w_list)
            flat_weights.append(self._flatten(w_list))

        # ── Step 2: reference centroid ────────────────────────────────────
        # Warmup  → robust geometric median (tolerates Byzantine outliers).
        # Normal  → lagged global model (cheap, adapts automatically).
        in_warmup = self.t_warmup is not None and self.current_round <= self.t_warmup
        if in_warmup or self.current_global_weights is None:
            ref = self._geometric_median(flat_weights)
            ref_label = "geometric_median"
        else:
            ref = self._flatten(self.current_global_weights)
            ref_label = "lagged_global"
        self.logger.info(f"Round {server_round}: centroid = {ref_label}")

        # ── Step 3: deviations Δᵢ = ||wᵢ - ref||₂ ───────────────────────
        deltas: Dict[str, float] = {
            cid: float(np.linalg.norm(flat - ref))
            for cid, flat in zip(client_ids, flat_weights)
        }

        # ── Step 4: discounted integral — decay all, then add participants ─
        # Spec: I ← I * λ  (for all clients)
        #       I[i] ← I[i] + Δᵢ  (for participants)
        for cid in list(self.client_integrals.keys()):
            self.client_integrals[cid] *= self.lam
        for cid in client_ids:
            self.client_integrals[cid] = self.client_integrals.get(cid, 0.0) + deltas[cid]

        # ── Step 5: PID scores Sᵢ = Kp·Δᵢ + Ki·I[i] + Kd·(Δᵢ - Δ_prev[i]) ─
        pid_scores: Dict[str, float] = {}
        for cid in client_ids:
            delta = deltas[cid]
            d_term = delta - self.client_delta_prev.get(cid, 0.0)
            pid_scores[cid] = (
                self.kp * delta
                + self.ki * self.client_integrals[cid]
                + self.kd * d_term
            )

        # ── Step 5a: update Delta_prev for participants ───────────────────
        for cid in client_ids:
            self.client_delta_prev[cid] = deltas[cid]

        # ── Step 6: threshold τₜ ─────────────────────────────────────────
        self.current_threshold = self._compute_threshold(list(pid_scores.values()))

        # ── Step 7: filter Q = {i : Sᵢ ≤ τₜ}; fallback to all if empty ──
        Q = [cid for cid in client_ids if pid_scores[cid] <= self.current_threshold]
        if not Q:
            self.logger.info(
                f"Round {server_round}: Q is empty — falling back to all {len(client_ids)} clients"
            )
            Q = list(client_ids)

        time_end = time.time_ns()

        # ── Record metrics ────────────────────────────────────────────────
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
            removal_threshold=self.current_threshold,
        )

        for cid in client_ids:
            score = pid_scores[cid]
            self.client_pid_scores[cid] = score
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=float(score),
                absolute_distance=float(deltas[cid]),
            )
            self.logger.info(
                f"Round {server_round} | Client {cid} | "
                f"Δ={deltas[cid]:.4f} | I={self.client_integrals[cid]:.4f} | "
                f"S={score:.4f} | τ={self.current_threshold:.4f} | in_Q={cid in Q}"
            )

        self.logger.info(
            f"Round {server_round}: τ={self.current_threshold:.4f} ({self.threshold_mode}) | "
            f"Q={Q} ({len(Q)}/{len(client_ids)} clients)"
        )

        # ── Step 8: aggregate over Q ──────────────────────────────────────
        q_results = [(proxy, res) for proxy, res in results if proxy.cid in Q]
        return super().aggregate_fit(server_round, q_results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info(
            '\n' + '-' * 50 + f' AGGREGATION ROUND {server_round} ' + '-' * 50
        )

        for client_result in results:
            cid = client_result[0].cid
            metrics = client_result[1].metrics
            metrics['cid'] = cid
            store_evaluation_metrics(
                clinet_id=int(cid),
                current_round=self.current_round,
                metrics=metrics, 
                strategy_history=self.strategy_history
            )

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_metadata.cid),
                current_round=self.current_round,
                loss=evaluate_res.loss,
            )
            if client_metadata.cid not in self.removed_client_ids:
                if self.strategy_history.strategy_config.model_type == "lstm":
                    aggregate_value.append((evaluate_res.metrics['num_elements'], evaluate_res.loss))
                else:
                    aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=self.removed_client_ids,
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.t_warmup,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f'Round: {server_round} | '
            f'Aggregated clients: {number_of_clients_in_loss_calc} | '
            f'Loss: {loss_aggregated:.4f}'
        )

        return loss_aggregated, metrics_aggregated
