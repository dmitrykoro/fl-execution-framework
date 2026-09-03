"""
LQR-Fed: LQR-Inspired Soft-Weight Aggregation.

Instead of applying a binary threshold to remove clients, LQR-Fed assigns a
continuous aggregation weight wᵢ ∈ [0, 1] to each client by solving a small
convex Quadratic Program (QP) each round:

    minimise   Σᵢ [ q·dᵢ²·wᵢ + r·(wᵢ − 1/N)² ]
    subject to Σ wᵢ = 1,   0 ≤ wᵢ ≤ 1

where dᵢ is the L2 distance of client i's weight vector from the reference
centroid, q is the state cost (penalty on trusting high-distance clients), and
r is the control cost (penalty on deviating from uniform weights).

The ratio q/r controls the trade-off:
  high q/r → aggressively downweight outliers
  low  q/r → stay close to standard FedAvg

Reference centroid:
  warmup rounds (t ≤ T_warmup): arithmetic mean of participant weight vectors
  post-warmup               : the lagged global model (weight vector sent to
                               clients at the start of that round)

No client is permanently excluded — the QP is re-solved from scratch each round.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics
from src.simulation_strategies.lqr_fed_scoring import (
    flatten_weights,
    compute_distances,
    solve_lqr_qp,
    weighted_aggregate,
)


class LQRFedStrategy(fl.server.strategy.FedAvg):
    """
    LQR-Fed aggregation strategy.

    Parameters
    ----------
    lqr_q : float
        State cost q > 0. Penalty on assigning aggregation weight to clients
        whose weight vector deviates far from the reference centroid.
    lqr_r : float
        Control cost r > 0. Penalty on deviating from uniform (equal)
        weights. High r keeps the distribution close to FedAvg.
    begin_removing_from_round : int
        Warmup period T_warmup. During rounds 1..T_warmup the reference
        centroid is the arithmetic mean of participant weight vectors.
        From round T_warmup+1 onward the lagged global model is used.
    remove_clients : bool
        Framework-level flag. LQR-Fed never hard-excludes clients, but when
        True the aggregation_participation is logged as 0 for any client
        whose QP weight rounds to 0 (i.e. below 1e-6).
    normalize_distances : bool
        When False (default, keyword ``lqr_fed``): raw L2 distances are fed
        to the QP, so q/r must be tuned per training regime.
        When True (keyword ``lqr_fed_norm_dist``): distances are divided by
        their round-wise mean before solving the QP.  This makes the
        effective q/r sensitivity invariant to local epoch count and data
        heterogeneity — a client twice as far as average always gets the
        same suppression regardless of absolute drift scale.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    """

    def __init__(
        self,
        lqr_q: float,
        lqr_r: float,
        begin_removing_from_round: int,
        remove_clients: bool,
        strategy_history: SimulationStrategyHistory,
        output_dirname: str,
        normalize_distances: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lqr_q = lqr_q
        self.lqr_r = lqr_r
        self.t_warmup = begin_removing_from_round
        self.remove_clients = remove_clients
        self.normalize_distances = normalize_distances
        self.strategy_history = strategy_history

        self.current_round: int = 0
        self.current_global_weights: Optional[List[np.ndarray]] = None

        self.logger = logging.getLogger(f"lqr_fed_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)
        os.makedirs(output_dirname, exist_ok=True)
        self.logger.addHandler(logging.FileHandler(f"{output_dirname}/output.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        # Capture the lagged global model before clients train on it
        self.current_global_weights = parameters_to_ndarrays(parameters)

        available_clients = client_manager.all()

        # LQR-Fed never hard-excludes: all clients always participate
        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=set()
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
            return None, {}

        time_start = time.time_ns()

        # ── Extract flattened weight vectors ──────────────────────────────
        client_ids: List[str] = []
        weight_lists: List[List[np.ndarray]] = []
        flat_weights: List[np.ndarray] = []

        for client_proxy, fit_res in results:
            wl = parameters_to_ndarrays(fit_res.parameters)
            client_ids.append(client_proxy.cid)
            weight_lists.append(wl)
            flat_weights.append(flatten_weights(wl))

        # ── Reference centroid ────────────────────────────────────────────
        in_warmup = self.t_warmup is not None and self.current_round <= self.t_warmup
        if in_warmup or self.current_global_weights is None:
            ref = np.mean(flat_weights, axis=0)
            ref_label = "arithmetic_mean"
        else:
            ref = flatten_weights(self.current_global_weights)
            ref_label = "lagged_global"

        # ── L2 distances (optionally mean-normalised) ─────────────────────
        distances = compute_distances(flat_weights, ref, normalize=self.normalize_distances)

        # ── Solve QP ─────────────────────────────────────────────────────
        qp_weights = solve_lqr_qp(distances, self.lqr_q, self.lqr_r)

        time_end = time.time_ns()

        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
        )

        # ── Log per-client metrics ────────────────────────────────────────
        effectively_zero_threshold = 1e-6
        for i, cid in enumerate(client_ids):
            w_i = float(qp_weights[i])
            d_i = float(distances[i])

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=w_i,
                absolute_distance=d_i,
                lqr_weight=w_i,
            )

            if self.remove_clients and w_i < effectively_zero_threshold:
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    aggregation_participation=0,
                )

            self.logger.info(
                f"Round {server_round} | Client {cid} | "
                f"d={d_i:.4f} | w={w_i:.4f} | ref={ref_label}"
            )

        norm_tag = " norm_dist=True" if self.normalize_distances else ""
        self.logger.info(
            f"Round {server_round}: QP weights={np.round(qp_weights, 4).tolist()} "
            f"(sum={qp_weights.sum():.6f}, ref={ref_label}{norm_tag})"
        )

        # ── Weighted aggregate ────────────────────────────────────────────
        aggregated = weighted_aggregate(weight_lists, qp_weights)
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info(
            "\n" + "-" * 50 + f" AGGREGATION ROUND {server_round} " + "-" * 50
        )

        for client_result in results:
            cid = client_result[0].cid
            metrics = client_result[1].metrics
            metrics["cid"] = cid
            # Dispatch on which keys the payload carries: flower_client.evaluate
            # returns accuracy for classification, {ss_res, ss_tot, r2} for
            # regression and {loss_baseline, mare, ...} for time-series models.
            store_evaluation_metrics(
                clinet_id=int(cid),
                current_round=self.current_round,
                metrics=metrics,
                strategy_history=self.strategy_history,
            )

        if not results:
            return None, {}

        aggregate_value = []
        num_aggregated = 0
        malicious_ids = self.strategy_history.get_malicious_client_ids()

        for client_metadata, evaluate_res in results:
            cid = int(client_metadata.cid)
            self.strategy_history.insert_single_client_history_entry(
                client_id=cid,
                current_round=self.current_round,
                loss=evaluate_res.loss,
            )
            if cid not in malicious_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                num_aggregated += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=set(),
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.t_warmup,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f"Round: {server_round} | "
            f"Aggregated clients: {num_aggregated} | "
            f"Loss: {loss_aggregated:.4f}"
        )

        return loss_aggregated, metrics_aggregated
