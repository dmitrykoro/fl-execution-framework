"""
FLTrust strategy (NDSS'21 / arXiv:2012.13995v3) for Flower.

Core idea (paper):
- Server computes a "server model update" g0 using a small clean root dataset.
- For each client update gi:
  - trust score: TS_i = ReLU(cos(gi, g0))
  - normalize magnitude: gbar_i = (||g0|| / ||gi||) * gi
- Aggregate: g = sum_i TS_i * gbar_i / sum_i TS_i
- Update global model: w <- w + alpha * g
"""

from __future__ import annotations

import time
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import SimulationStrategyHistory

NDArrayList = List[np.ndarray]


# ---------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------
def _flatten(nds: NDArrayList) -> np.ndarray:
    """Flatten list of ndarrays into one 1D vector (float64 for stable dot/norm)."""
    if not nds:
        return np.array([], dtype=np.float64)
    return np.concatenate([x.astype(np.float64, copy=False).ravel() for x in nds])


def _l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _add(a: NDArrayList, b: NDArrayList) -> NDArrayList:
    return [x + y for x, y in zip(a, b)]


def _sub(a: NDArrayList, b: NDArrayList) -> NDArrayList:
    return [x - y for x, y in zip(a, b)]


def _scale(a: NDArrayList, s: float) -> NDArrayList:
    return [x * s for x in a]


def _zeros_like(a: NDArrayList) -> NDArrayList:
    return [np.zeros_like(x) for x in a]


# ---------------------------------------------------------------------
# FLTrust Strategy
# ---------------------------------------------------------------------
class FLTrustStrategy(fl.server.strategy.FedAvg):
    """
    FLTrust aggregation strategy.

    You MUST provide `server_update_fn`, because the strategy needs g0, the
    "server model update" computed on the root dataset (paper's Step II). :contentReference[oaicite:5]{index=5}

    server_update_fn signature:
        g0 = server_update_fn(global_params_ndarrays, server_round)
    where g0 is a list of ndarrays of the same shapes as model parameters, representing an update (delta).

    Notes:
    - This strategy does not inherently *remove* clients; it *downweights* them via TS_i.
    - It stores TS and cosine in SimulationStrategyHistory, reusing your existing fields.
    """

    def __init__(
            self,
            *,
            server_update_fn: Callable[[NDArrayList, int], NDArrayList],
            global_lr: float = 1.0,  # paper's alpha :contentReference[oaicite:6]{index=6}
            strategy_history: Optional[SimulationStrategyHistory] = None,
            log_trust_scores: bool = True,
            eps: float = 1e-12,
            # FedAvg args/kwargs
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.server_update_fn = server_update_fn
        self.global_lr = float(global_lr)
        self.strategy_history = strategy_history
        self.log_trust_scores = log_trust_scores
        self.eps = float(eps)

        self.current_round: int = 0
        self._last_global_params: Optional[NDArrayList] = None

        self.logger = logging.getLogger(f"fltrust_{id(self)}")
        self.logger.setLevel(logging.INFO)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Save current global params so we can compute client updates gi = wi - w in aggregate_fit
        self._last_global_params = fl.common.parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        # Fall back to FedAvg if nothing returned
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        if self._last_global_params is None:
            # Defensive: if configure_fit wasn't called for some reason
            self._last_global_params = fl.common.parameters_to_ndarrays(results[0][1].parameters)

        w_global = self._last_global_params

        # -----------------------------------------------------------------
        # Step II (paper): compute server model update g0 using root dataset
        # -----------------------------------------------------------------
        g0 = self.server_update_fn(w_global, server_round)

        g0_vec = _flatten(g0)
        norm_g0 = _l2_norm(g0_vec)

        # If g0 is (near) zero, FLTrust can't define directions/magnitudes well.
        # We return FedAvg in that case (safe fallback).
        if norm_g0 < self.eps:
            self.logger.warning("FLTrust: ||g0|| too small; falling back to FedAvg for this round.")
            return super().aggregate_fit(server_round, results, failures)

        # -----------------------------------------------------------------
        # Step III (paper): trust scores + normalization + weighted average
        # Trust score TS_i = ReLU(cos(gi, g0)) :contentReference[oaicite:7]{index=7}
        # Normalize magnitudes to ||g0|| :contentReference[oaicite:8]{index=8}
        # -----------------------------------------------------------------
        time_start = time.time_ns()

        weighted_sum = _zeros_like(g0)
        sum_ts = 0.0

        for client_proxy, fit_res in results:
            cid = int(client_proxy.cid)
            w_client = fl.common.parameters_to_ndarrays(fit_res.parameters)

            gi = _sub(w_client, w_global)  # client model update
            gi_vec = _flatten(gi)

            cosine = _cosine_similarity(gi_vec, g0_vec, eps=self.eps)
            ts = max(0.0, cosine)  # ReLU clip

            # Normalize update magnitude to ||g0||:
            # gbar_i = (||g0||/||gi||) * gi
            norm_gi = _l2_norm(gi_vec)
            if norm_gi < self.eps:
                gbar_i = _zeros_like(gi)
            else:
                gbar_i = _scale(gi, norm_g0 / norm_gi)

            # Trust-weighted accumulation
            if ts > 0.0:
                weighted_sum = _add(weighted_sum, _scale(gbar_i, ts))
                sum_ts += ts

            # Store into your strategy history (reuse existing fields)
            if self.strategy_history is not None:
                # removal_criterion_history is used as a "score" field in your framework
                # (we store TS_i), and absolute_distance_history can store cosine.
                self.strategy_history.insert_single_client_history_entry(
                    client_id=cid,
                    current_round=self.current_round,
                    removal_criterion=float(ts),
                    absolute_distance=float(cosine),
                )

            if self.log_trust_scores:
                self.logger.info(
                    f"FLTrust round={server_round} cid={cid} cosine={cosine:.6f} TS={ts:.6f} "
                    f"num_examples={fit_res.num_examples}"
                )

        # Compute global model update g
        if sum_ts < self.eps:
            # Nobody aligned with server update direction; apply no update.
            g = _zeros_like(g0)
        else:
            g = _scale(weighted_sum, 1.0 / sum_ts)

        # Apply global learning rate alpha: w <- w + alpha * g :contentReference[oaicite:9]{index=9}
        w_new = _add(w_global, _scale(g, self.global_lr))

        time_end = time.time_ns()
        if self.strategy_history is not None:
            self.strategy_history.insert_round_history_entry(
                score_calculation_time_nanos=int(time_end - time_start)
            )

        # Return new global parameters to Flower server
        return ndarrays_to_parameters(w_new), {}
