"""
FLDetector: Defending Federated Learning Against Model Poisoning Attacks
via Detecting Malicious Clients.

Reference
---------
Zhang, Z., Cao, X., Jia, J., & Gong, N. Z.
"FLDetector: Defending Federated Learning Against Model Poisoning Attacks
via Detecting Malicious Clients."
ACM KDD 2022.
https://dl.acm.org/doi/10.1145/3534678.3539231
https://arxiv.org/abs/2207.09209

Algorithm Overview
------------------
FLDetector is a *pre-filtering* layer that wraps a standard aggregator
(FedAvg in this implementation).  Each round the server:

  1. Predicts what every client's update *should* be this round using a
     quasi-Newton (L-BFGS) estimate of the local curvature built from
     the last ``history_size`` rounds of gradient-update pairs (s, y).
     During the warmup period (round ≤ ``begin_removing_from_round``) the
     L-BFGS history is not yet populated, so the prediction falls back to
     the zero vector (i.e. the previous global model is returned as-is
     and no filtering is performed).

  2. Measures the consistency of each client's actual update with its
     prediction using the L2 distance:
         inconsistency_i^t = ||Δw_i^t  −  Δŵ_i^t||₂

  3. Accumulates a per-client *suspicious score* that is the sum of
     inconsistency scores over all rounds since the last reset:
         S_i^t = Σ_{τ=1}^{t} inconsistency_i^τ

  4. Flags clients whose suspicious score exceeds a threshold derived from
     the score distribution (mean + ``z_score_threshold`` × std).
     Flagged clients are excluded from the FedAvg aggregation step.

Key hyperparameters (from the paper)
-------------------------------------
- history_size (N):
    Window of L-BFGS curvature pairs retained.  The paper uses N = 10.
    Larger N gives better curvature approximation at the cost of O(N·d)
    memory per client.

- z_score_threshold:
    Number of standard deviations above the population mean used to set
    the removal threshold.  The paper does not fix a single value; values
    in [1.0, 3.0] are reasonable.  Lower = more aggressive removal
    (higher FP risk); higher = more conservative (higher FN risk).
    The paper's experiments typically show bimodal score separation by
    round ~60, so at 30 rounds a lower threshold (1.5–2.0) is recommended
    to catch malicious clients before full separation occurs.

- begin_removing_from_round:
    Warmup period.  The paper reports that the suspicious-score
    distributions become clearly bimodal around round 60 for 50-client
    settings.  For 100-client settings with 30 rounds this implies that
    filtering should start early (round 7–10) and a moderately aggressive
    threshold should be used, because full score separation will not be
    reached within the experiment.

Computational complexity
------------------------
- Per round: O(n · N · d) — N L-BFGS two-loop recursion steps per client,
  each step O(d).  With N = 10 this is approximately 10× heavier than
  PID-MADE's O(n · d).
- Memory: O(n · N · d) — history_size gradient-update pairs per client.
  For large models (d ≫ 10⁶) and many clients this can be substantial.

Known limitations
-----------------
1. Requires a burn-in period (~60 rounds per the paper) for score
   distributions to become bimodal; at 30 rounds detection is noisier.
2. The L-BFGS approximation degrades under strong non-IID data
   (α ≤ 0.1) because the global gradient history poorly approximates
   each client's local curvature.
3. This is a two-stage design: FLDetector pre-filters, then FedAvg
   aggregates the remaining clients.  The paper evaluated with Krum /
   Trimmed-mean as the second stage; FedAvg is used here for simplicity
   and symmetry with the other strategies.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics


class FLDetectorStrategy(fl.server.strategy.FedAvg):
    """
    FLDetector aggregation strategy (KDD 2022).

    Parameters
    ----------
    history_size : int
        Number of L-BFGS curvature pairs (s, y) retained per client
        (paper default: 10).  Must be ≥ 1.
    z_score_threshold : float
        Removal threshold expressed as standard deviations above the
        per-round mean suspicious score.  Clients with score >
        (μ + z × σ) are excluded from aggregation.
        Recommended range: 1.5–2.5.  Lower values are more aggressive.
    remove_clients : bool
        Whether to actually exclude flagged clients from aggregation.
        Set to False to collect scores without modifying aggregation
        (useful for debugging / analysis).
    begin_removing_from_round : int
        Warmup rounds before filtering is activated.  During warmup,
        L-BFGS history is still accumulated, but no clients are removed.
        Recommended: 7–10 (matching other strategies in this framework).
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    output_dirname : str
        Directory for log output.
    """

    def __init__(
        self,
        history_size: int,
        z_score_threshold: float,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        output_dirname: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if history_size < 1:
            raise ValueError(f"history_size must be ≥ 1, got {history_size}")
        if z_score_threshold <= 0:
            raise ValueError(f"z_score_threshold must be > 0, got {z_score_threshold}")

        self.history_size = history_size
        self.z_score_threshold = z_score_threshold
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.strategy_history = strategy_history

        # ── Per-round state ────────────────────────────────────────────────
        self.current_round: int = 0
        self.removed_client_ids: Set[str] = set()

        # Previous global model parameters (flat numpy vector).
        # Stored at the end of configure_fit so that aggregate_fit can
        # compute the true update delta = w_client - w_global_prev.
        self._prev_global_flat: Optional[np.ndarray] = None

        # Per-client L-BFGS curvature history.
        # _lbfgs_s[cid] = deque of s vectors (update in parameter space)
        # _lbfgs_y[cid] = deque of y vectors (change in gradient estimate)
        # We use plain lists and truncate to history_size.
        self._lbfgs_s: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._lbfgs_y: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Per-client suspicious score accumulator.
        self._suspicious_scores: Dict[str, float] = defaultdict(float)

        # Previous round's gradient estimates (for computing y = g_new - g_old).
        # g_estimate[cid] = flat gradient estimate used in the PREVIOUS round.
        self._prev_grad_estimate: Dict[str, np.ndarray] = {}

        # Logger
        self.logger = logging.getLogger(f"fl_detector_{id(self)}")
        self.logger.setLevel(logging.INFO)
        os.makedirs(output_dirname, exist_ok=True)
        self.logger.addHandler(logging.FileHandler(f"{output_dirname}/output.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # L-BFGS two-loop recursion
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(weights: List[np.ndarray]) -> np.ndarray:
        """Concatenate all weight arrays into a single float64 vector."""
        return np.concatenate([w.flatten().astype(np.float64) for w in weights])

    def _lbfgs_product(self, cid: str, gradient: np.ndarray) -> np.ndarray:
        """
        Compute H⁻¹ · gradient using the L-BFGS two-loop recursion.

        This approximates the inverse Hessian-gradient product using the
        stored (s, y) curvature pairs for client ``cid``.

        If no history is available the identity matrix is used (i.e.
        gradient descent step).

        Returns
        -------
        np.ndarray
            The (approximate) Newton step direction.
        """
        s_list = self._lbfgs_s[cid]
        y_list = self._lbfgs_y[cid]

        if not s_list:
            # No history yet — fall back to gradient (identity Hessian)
            return gradient.copy()

        m = len(s_list)
        q = gradient.copy()
        alphas = np.zeros(m)

        # First loop (most recent pairs first)
        for i in reversed(range(m)):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            alpha_i = rho_i * np.dot(s_i, q)
            alphas[i] = alpha_i
            q -= alpha_i * y_i

        # Initial Hessian scaling: H₀ = (sᵀy / yᵀy) · I  (Nocedal & Wright eq. 7.20)
        s_last = s_list[-1]
        y_last = y_list[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        r = gamma * q

        # Second loop
        for i in range(m):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            beta_i = rho_i * np.dot(y_i, r)
            r += s_i * (alphas[i] - beta_i)

        return r

    def _predict_update(self, cid: str, current_grad_estimate: np.ndarray) -> np.ndarray:
        """
        Predict client cid's update this round using L-BFGS.

        The Cauchy Mean Value Theorem + L-BFGS approximation gives:
            Δŵ_i^t ≈ H_i^{-1} · g_i^t
        where g_i^t is the current gradient estimate and H_i is the
        approximate Hessian built from stored (s, y) pairs.

        During the first round (or if no history exists) this returns the
        zero vector (no prediction available).
        """
        if cid not in self._prev_grad_estimate or not self._lbfgs_s[cid]:
            return np.zeros_like(current_grad_estimate)

        return self._lbfgs_product(cid, current_grad_estimate)

    def _update_lbfgs_history(
        self,
        cid: str,
        delta_params: np.ndarray,
        current_grad_estimate: np.ndarray,
    ) -> None:
        """
        Update L-BFGS history for client cid.

        s = Δw  (change in parameters = client update this round)
        y = g_new - g_old  (change in gradient estimate)

        Maintains a sliding window of ``history_size`` pairs.
        """
        if cid in self._prev_grad_estimate:
            s = delta_params.copy()
            y = current_grad_estimate - self._prev_grad_estimate[cid]

            # Only add if curvature condition holds (yᵀs > 0) to keep H⁻¹ PD
            if np.dot(y, s) > 1e-10:
                self._lbfgs_s[cid].append(s)
                self._lbfgs_y[cid].append(y)
                if len(self._lbfgs_s[cid]) > self.history_size:
                    self._lbfgs_s[cid].pop(0)
                    self._lbfgs_y[cid].pop(0)

        self._prev_grad_estimate[cid] = current_grad_estimate.copy()

    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Record the current global model for computing deltas in aggregate_fit,
        then send parameters to all available clients.
        """
        # Store flat global model before clients train
        global_weights = fl.common.parameters_to_ndarrays(parameters)
        self._prev_global_flat = self._flatten(global_weights)

        available_clients = client_manager.all()

        # Carry forward removed clients from previous round into participation log
        self.strategy_history.update_client_participation(
            current_round=server_round,
            removed_client_ids=self.removed_client_ids,
        )

        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(client, fit_ins) for client in available_clients.values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        time_start = time.time_ns()

        in_warmup = (
            self.begin_removing_from_round is not None
            and self.current_round <= self.begin_removing_from_round
        )

        # ── Step 1: Compute client deltas and gradient estimates ───────────
        # delta_i = w_client_i − w_global_prev  (the actual update)
        # gradient_estimate_i ≈ delta_i  (we use the raw update as the
        # gradient proxy; the paper uses (w_global - w_client) / η_eff,
        # but since η_eff is absorbed into the delta we use delta directly)

        client_ids: List[str] = []
        deltas: List[np.ndarray] = []
        weight_lists: List[List[np.ndarray]] = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_flat = self._flatten(client_weights)

            if self._prev_global_flat is not None:
                delta = client_flat - self._prev_global_flat
            else:
                delta = np.zeros_like(client_flat)

            client_ids.append(cid)
            deltas.append(delta)
            weight_lists.append(client_weights)

        # ── Step 2: Predict expected updates via L-BFGS ───────────────────
        predictions: List[np.ndarray] = []
        for cid, delta in zip(client_ids, deltas):
            # Gradient estimate for this client this round
            grad_estimate = delta.copy()
            pred = self._predict_update(cid, grad_estimate)
            predictions.append(pred)
            # Update L-BFGS history with this round's data
            self._update_lbfgs_history(cid, delta, grad_estimate)

        # ── Step 3: Compute inconsistency scores ──────────────────────────
        inconsistencies: List[float] = []
        for delta, pred in zip(deltas, predictions):
            score = float(np.linalg.norm(delta - pred))
            inconsistencies.append(score)

        # ── Step 4: Accumulate suspicious scores ─────────────────────────
        for cid, inc in zip(client_ids, inconsistencies):
            self._suspicious_scores[cid] += inc

        suspicious_values = np.array([self._suspicious_scores[cid] for cid in client_ids])

        # ── Step 5: Compute removal threshold ────────────────────────────
        mu = float(np.mean(suspicious_values))
        sigma = float(np.std(suspicious_values))
        threshold = mu + self.z_score_threshold * sigma

        time_end = time.time_ns()

        # ── Step 6: Flag clients above threshold ──────────────────────────
        if in_warmup or not self.remove_clients:
            self.removed_client_ids = set()
        else:
            self.removed_client_ids = {
                cid
                for cid, score in zip(client_ids, suspicious_values)
                if score > threshold
            }

        accepted_results = [
            (proxy, res)
            for (proxy, res) in results
            if proxy.cid not in self.removed_client_ids
        ]

        if not accepted_results:
            self.logger.info(
                f"Round {server_round}: all clients flagged — falling back to all"
            )
            accepted_results = results
            self.removed_client_ids = set()

        # ── Record metrics ────────────────────────────────────────────────
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
            removal_threshold=threshold,
        )
        self.strategy_history.update_client_participation(
            current_round=server_round,
            removed_client_ids=self.removed_client_ids,
        )

        for cid, score in zip(client_ids, suspicious_values):
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=float(score),
                absolute_distance=float(
                    inconsistencies[client_ids.index(cid)]
                ),
            )

        # ── Logging ───────────────────────────────────────────────────────
        for cid in self.removed_client_ids:
            self.logger.info(
                f"Round {server_round}: flagged Client {cid} "
                f"(score={self._suspicious_scores[cid]:.4f}, threshold={threshold:.4f})"
            )

        self.logger.info(
            f"Round {server_round}: "
            f"accepted {len(accepted_results)}/{len(results)} clients | "
            f"threshold={threshold:.4f} | "
            f"μ={mu:.4f} | σ={sigma:.4f}"
        )

        # ── Step 7: FedAvg over accepted clients ──────────────────────────
        return super().aggregate_fit(server_round, accepted_results, failures)

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
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=float(metrics.get('accuracy', 0.0)),
            )

        if not results:
            return None, {}

        aggregate_value = []
        num_aggregated = 0

        for client_metadata, evaluate_res in results:
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_metadata.cid),
                current_round=self.current_round,
                loss=evaluate_res.loss,
            )
            if client_metadata.cid not in self.removed_client_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                num_aggregated += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=self.removed_client_ids,
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.begin_removing_from_round,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f'Round: {server_round} | '
            f'Aggregated clients: {num_aggregated} | '
            f'Loss: {loss_aggregated:.4f}'
        )

        return loss_aggregated, metrics_aggregated
