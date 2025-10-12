import time
import numpy as np
import flwr as fl
import torch
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

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

from src.output_handlers.directory_handler import DirectoryHandler
from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class BulyanStrategy(fl.server.strategy.FedAvg):
    """Bulyan aggregation strategy implemented in the *same coding style* as
    the provided ``MultiKrumStrategy``.

    Arguments mirror the original class so the surrounding secure‑FL
    framework can swap strategies without refactoring.
    """

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(
        self,
        remove_clients: bool,
        num_krum_selections: int, # n - f
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # --- Public / framework‑visible fields -------------------------
        self.remove_clients = remove_clients
        self.num_krum_selections = num_krum_selections # n - f
        self.begin_removing_from_round = begin_removing_from_round

        # --- Internal state -------------------------------------------
        self.client_scores: Dict[str, float] = {}
        self.removed_client_ids: set[str] = set()
        self.current_round: int = 0
        self.strategy_history = strategy_history

        # --- Logger (matches MultiKrum style) -------------------------
        self.logger = logging.getLogger(f"bulyan_{id(self)}")
        self.logger.setLevel(logging.INFO)
        out_dir = DirectoryHandler.dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # Helper: pairwise distances (cached) ------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
        """Return condensed Euclidean distance matrix squared."""
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.square(np.linalg.norm(diff, axis=2))

    # ------------------------------------------------------------------
    # Core aggregation --------------------------------------------------
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return None, {}

        # ---------------- Clustering diagnostics (optional) ------------
        clustering_param_data = []
        for _, fit_res in results:
            tensors = [torch.tensor(arr).flatten() for arr in parameters_to_ndarrays(fit_res.parameters)]
            clustering_param_data.append(torch.cat(tensors))
        X_embed = np.vstack([t.numpy() for t in clustering_param_data])
        kmeans = KMeans(n_clusters=1, init="k-means++").fit(X_embed)
        abs_distances = kmeans.transform(X_embed)
        norm_distances = MinMaxScaler().fit(abs_distances).transform(abs_distances)

        # ---------------- Flatten updates ------------------------------
        param_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        flat_updates = np.stack([np.concatenate([p.ravel() for p in pa]) for pa in param_arrays])
        n, dim = flat_updates.shape
        C = self.num_krum_selections
        f = (n - C) // 2  # number of malicious clients
        if C > n or (n - C) % 2:
            self.logger.error("C must satisfy n - C = 2f (even, <= n)")
            return super().aggregate_fit(server_round, results, failures)

        # Ensure Bulyan preconditions
        if n <= 4 * f + 2:
            self.logger.warning(
                f"[Bulyan] Not enough clients ({n}) for assumed f={f}. Using simple mean."
            )
            return super().aggregate_fit(server_round, results, failures)

        # ----------------- Multi‑Krum phase ----------------------------
        time_start_calc = time.time_ns()
        dists = self._pairwise_sq_dists(flat_updates)
        m = n - f - 2  # number of nearest neighbours to sum
        krum_scores = np.array([np.partition(dists[i], m)[:m].sum() for i in range(n)])
        candidate_idx = np.argpartition(krum_scores, C)[:C]
        candidates = flat_updates[candidate_idx]

        # ----------------- Trimmed‑mean phase --------------------------
        # drop top f and bottom f scores from C candidates
        sorted_idx = np.argsort(candidates, axis=0)
        kept_slice = slice(f, C - f)
        trimmed = candidates[sorted_idx[kept_slice, np.arange(dim)], np.arange(dim)]
        bulyan_vector = trimmed.mean(axis=0)
        time_end_calc = time.time_ns()

        # ----------------- Prepare return params -----------------------
        agg_list, cursor = [], 0
        for arr in param_arrays[0]:
            num = arr.size
            agg_list.append(bulyan_vector[cursor : cursor + num].reshape(arr.shape).astype(arr.dtype))
            cursor += num
        aggregated_parameters = ndarrays_to_parameters(agg_list)

        # ----------------- Book‑keeping & logging ----------------------
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end_calc - time_start_calc
        )

        for i, (client_proxy, _) in enumerate(results):
            cid = client_proxy.cid
            deviation = float(np.linalg.norm(flat_updates[i] - bulyan_vector))
            # if neeeded krum scores instead uncomment the following line
            # deviation = float(krum_scores[i])
            self.client_scores[cid] = deviation
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=deviation,
                absolute_distance=float(abs_distances[i][0]),
            )
            self.logger.info(
                f"Aggregation round: {server_round} Client ID: {cid} Deviation: {deviation:.4e} "
                f"Normalized Distance: {norm_distances[i][0]:.4f}"
            )

        return aggregated_parameters, {}

    # ------------------------------------------------------------------
    # Client selection --------------------------------------------------
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        available_clients = client_manager.all()

        # Warm‑up: keep everyone
        if self.current_round <= self.begin_removing_from_round:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(c, fit_ins) for c in available_clients.values()]

        # --- Gather scores for available clients ----------------------
        client_scores = {cid: self.client_scores.get(cid, 0.0) for cid in available_clients.keys()}

        # --- Reset removed set each round -----------------------------
        self.removed_client_ids = set()

        if self.remove_clients:
            # Remove *f* clients with highest scores this round
            n = len(client_scores)
            f = max(0, n - self.num_krum_selections) // 2
            for _ in range(f):
                if not client_scores:
                    break
                eligible = {cid: s for cid, s in client_scores.items() if cid not in self.removed_client_ids}
                if not eligible:
                    break
                worst = max(eligible, key=eligible.get)
                self.removed_client_ids.add(worst)

        self.logger.info(
            f"Removed clients at round {self.current_round} are : {self.removed_client_ids}"
        )

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )

        # --- Build fit instructions ----------------------------------
        ordered_cids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {})
        return [
            (available_clients[cid], fit_ins)
            for cid in ordered_cids
            if cid in available_clients
        ]

    # ------------------------------------------------------------------
    # Evaluation aggregation (identical pattern to Multi‑Krum) ---------
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info("\n" + "-" * 50 + f"AGGREGATION ROUND {server_round}" + "-" * 50)

        for cp, ev in results:
            cid = cp.cid
            acc_metrics = dict(ev.metrics)
            acc_metrics["cid"] = cid
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=acc_metrics.get("accuracy"),
            )

        if not results:
            return None, {}

        aggregate_value, num_clients_loss = [], 0
        for cp, ev in results:
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cp.cid), current_round=self.current_round, loss=ev.loss
            )
            if cp.cid not in self.removed_client_ids:
                aggregate_value.append((ev.num_examples, ev.loss))
                num_clients_loss += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

                # --------- OPTIONAL strict MedMentions micro-averaged metrics ----------
        # Only enable if at least one client reported strict counts (NER/GPT-2 case)
        has_strict = any(
            ("tp_m" in ev.metrics) or ("tp_d" in ev.metrics) for _, ev in results
        )

        metrics_aggregated: Dict[str, Scalar] = {}  # keep existing dict if you already have one
        if self.remove_clients and self.current_round >= self.begin_removing_from_round:
            effective_resullts = [(cp,ev) for (cp,ev) in results if cp.cid not in self.removed_client_ids]
        else:
            effective_resullts = results
        if has_strict:
            sum_tp_m = sum(int(ev.metrics.get("tp_m", 0)) for _, ev in effective_resullts)
            sum_fp_m = sum(int(ev.metrics.get("fp_m", 0)) for _, ev in effective_resullts)
            sum_fn_m = sum(int(ev.metrics.get("fn_m", 0)) for _, ev in effective_resullts)

            sum_tp_d = sum(int(ev.metrics.get("tp_d", 0)) for _, ev in effective_resullts)
            sum_fp_d = sum(int(ev.metrics.get("fp_d", 0)) for _, ev in effective_resullts)
            sum_fn_d = sum(int(ev.metrics.get("fn_d", 0)) for _, ev in effective_resullts)

            def _prf(tp: int, fp: int, fn: int):
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
                return p, r, f1

            mp, mr, mf1 = _prf(sum_tp_m, sum_fp_m, sum_fn_m)
            dp, dr, df1 = _prf(sum_tp_d, sum_fp_d, sum_fn_d)

            # Persist to rounds history only if those fields exist (step 5 adds them)
            rh = getattr(self.strategy_history, "rounds_history", None)
            if rh and hasattr(rh, "mention_precision_history"):
                rh.mention_precision_history.append(mp)
                rh.mention_recall_history.append(mr)
                rh.mention_f1_history.append(mf1)
                rh.document_precision_history.append(dp)
                rh.document_recall_history.append(dr)
                rh.document_f1_history.append(df1)

            # Expose in aggregated metrics output (nice for logs/exports)
            metrics_aggregated.update({
                "mention_precision": float(mp),
                "mention_recall": float(mr),
                "mention_f1": float(mf1),
                "document_precision": float(dp),
                "document_recall": float(dr),
                "document_f1": float(df1),
            })
        # -----------------------------------------------------------------------


        for cp, ev in results:
            logging.debug(f"Client ID: {cp.cid} Metrics: {ev.metrics} Loss: {ev.loss}")

        self.logger.info(
            f"Round: {server_round} Number of aggregated clients: {num_clients_loss} "
            f"Aggregated loss: {loss_aggregated}"
        )
        if has_strict:
            return loss_aggregated, metrics_aggregated
        return loss_aggregated, {}
