import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

# Optional: remove if you do not need torch‑based clustering diagnostics
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Project‑specific history tracker
from data_models.simulation_strategy_history import SimulationStrategyHistory


class BulyanStrategy(fl.server.strategy.FedAvg):
    """Bulyan aggregation strategy compatible with the secure‑FL framework.

    Two‑stage robust aggregation (Mhamdi *et al.*, ICML 2018):
    1. **Multi‑Krum** chooses `n − 2f` candidates most mutually consistent.
    2. **Coordinate‑wise trimmed mean** discards the largest & smallest *f*
       values per coordinate among those candidates.

    The class preserves the same public attributes (`current_round`,
    `rounds_history`, `client_scores`, etc.) and logging style used by the
    previously implemented `RFABasedRemovalStrategy` so downstream modules
    (dashboards, removal logic, history serialization) require no changes.
    """

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        network_model,
        aggregation_strategy_keyword: str,
        assumed_num_malicious: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.strategy_history = strategy_history
        self.network_model = network_model
        self.aggregation_strategy_keyword = aggregation_strategy_keyword
        self.assumed_num_malicious = assumed_num_malicious  # 'f' in Bulyan

        # Internal state matching the framework conventions
        self.current_round: int = 0
        self.removed_client_ids: set[str] = set()
        self.rounds_history: Dict[str, Dict] = {}
        self.client_scores: Dict[str, float] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Core FedAvg hooks -------------------------------------------------
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate parameters using **Bulyan** (Multi‑Krum + trimmed mean)."""
        self.current_round += 1
        self.rounds_history[str(self.current_round)] = {"client_info": {}, "round_info": {}}

        if not results:
            self.logger.warning("[Bulyan] No successful client results in round %s", server_round)
            return None, {}

        # Exclude permanently removed clients
        active = [(cp, fr) for cp, fr in results if cp.cid not in self.removed_client_ids]
        n = len(active)
        f = self.assumed_num_malicious
        if n <= 4 * f + 2:
            self.logger.warning(
                "[Bulyan] Not enough clients (%s) for assumed f=%s; falling back to mean.",
                n,
                f,
            )
            return super().aggregate_fit(server_round, results, failures)

        # 1. Flatten updates
        param_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in active]
        flats = np.stack([np.concatenate([p.ravel() for p in pa]) for pa in param_arrays])
        dim = flats.shape[1]

        # 2. Multi‑Krum pre‑selection
        distances = np.square(np.linalg.norm(flats[:, None, :] - flats[None, :, :], axis=2))
        m = n - f - 2  # number of closest vectors to sum for each score
        krum_scores = np.array([np.partition(distances[i], m)[:m].sum() for i in range(n)])
        sel_cnt = n - 2 * f
        selected_idx = np.argpartition(krum_scores, sel_cnt)[:sel_cnt]
        candidates = flats[selected_idx]

        # 3. Coordinate‑wise trimmed mean
        sorted_idx = np.argsort(candidates, axis=0)
        keep_slice = slice(f, sel_cnt - f)
        kept = candidates[sorted_idx[keep_slice, np.arange(dim)], np.arange(dim)]
        bulyan_vec = kept.mean(axis=0)

        # 4. Per‑client deviation bookkeeping (optional diagnostics)
        for idx, (cp, _) in enumerate(active):
            cid = cp.cid
            dev = float(np.linalg.norm(flats[idx] - bulyan_vec))
            self.client_scores[cid] = dev
            self.rounds_history[str(self.current_round)]["client_info"][f"client_{cid}"] = {
                "removal_criterion": dev,
                "is_removed": self.rounds_history
                .get(str(self.current_round - 1), {})
                .get("client_info", {})
                .get(f"client_{cid}", {})
                .get("is_removed", False),
            }

        # Optional cluster distance monitoring (same pattern as RFA, but skip if deps missing)
        try:
            tensors = [torch.cat([torch.tensor(a).flatten() for a in pa]) for pa in param_arrays]
            X = np.vstack([t.numpy() for t in tensors])
            raw_dist = KMeans(n_clusters=1, init="k-means++").fit(X).transform(X)
            norm_dist = MinMaxScaler().fit(raw_dist).transform(raw_dist)
            for idx, (cp, _) in enumerate(active):
                cid = cp.cid
                self.rounds_history[str(self.current_round)]["client_info"][f"client_{cid}"] |= {
                    "absolute_distance": float(raw_dist[idx][0]),
                    "normalized_distance": float(norm_dist[idx][0]),
                }
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("[Bulyan] Diagnostics skipped: %s", exc)

        # 5. Re‑assemble aggregated parameters
        out, cur = [], 0
        for arr in param_arrays[0]:
            num = arr.size
            out.append(bulyan_vec[cur : cur + num].reshape(arr.shape).astype(arr.dtype))
            cur += num
        return ndarrays_to_parameters(out), {}

    # ------------------------------------------------------------------
    # Client selection & evaluation
    # ------------------------------------------------------------------

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        available = client_manager.all()
        if self.current_round <= self.begin_removing_from_round:
            return [(c, fl.common.FitIns(parameters, {})) for c in available.values()]
        scores = {cid: self.client_scores.get(cid, 0.0) for cid in available}
        if self.remove_clients and scores:
            worst = max(scores, key=scores.get)
            self.logger.info("[Bulyan] Removing client %s (highest deviation)", worst)
            self.removed_client_ids.add(worst)
            self.rounds_history[str(self.current_round)]["client_info"][f"client_{worst}"]["is_removed"] = True
        ordered = sorted(scores, key=scores.get, reverse=True)
        return [(available[c], fl.common.FitIns(parameters, {})) for c in ordered if c in available]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Re‑use FedAvg’s default evaluation aggregation (or your custom one
        # if you copy the identical logic here). This keeps dashboards aligned.
        return super().aggregate_evaluate(server_round, results, failures)
