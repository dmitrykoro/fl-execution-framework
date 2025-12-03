import time
import numpy as np
import flwr as fl
import torch
import logging
import os
from typing import Optional, Union

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
    """Bulyan Byzantine-resilient aggregation strategy.

    Combines Multi-Krum selection with coordinate-wise trimmed mean to filter
    malicious client updates before aggregation.
    """

    def __init__(
        self,
        remove_clients: bool,
        num_krum_selections: int,  # n - f
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.remove_clients = remove_clients
        self.num_krum_selections = num_krum_selections
        self.begin_removing_from_round = begin_removing_from_round
        self.client_scores: dict[str, float] = {}
        self.removed_client_ids: set[str] = set()
        self.current_round: int = 0
        self.strategy_history = strategy_history
        self.logger = logging.getLogger(f"bulyan_{id(self)}")
        self.logger.setLevel(logging.INFO)
        out_dir = DirectoryHandler.dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    @staticmethod
    def _pairwise_sq_dists(vectors: np.ndarray) -> np.ndarray:
        """Return condensed Euclidean distance matrix squared."""
        diff = vectors[:, None, :] - vectors[None, :, :]
        return np.square(np.linalg.norm(diff, axis=2))

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate client model updates using the Bulyan algorithm.

        Applies Multi-Krum to select candidate clients, then computes
        coordinate-wise trimmed mean for Byzantine-resilient aggregation.

        Args:
            server_round: Current round number from the Flower server.
            results: List of (ClientProxy, FitRes) tuples from clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of (aggregated parameters, metrics dict).
        """
        self.current_round += 1

        # Update client.is_malicious based on attack_schedule for dynamic attacks
        if self.strategy_history:
            self.strategy_history.update_client_malicious_status(server_round)

        if not results:
            return None, {}

        clustering_param_data = []
        for _, fit_res in results:
            tensors = [
                torch.tensor(arr).flatten()
                for arr in parameters_to_ndarrays(fit_res.parameters)
            ]
            clustering_param_data.append(torch.cat(tensors))
        X_embed = np.vstack([t.numpy() for t in clustering_param_data])
        kmeans = KMeans(n_clusters=1, init="k-means++").fit(X_embed)
        abs_distances = kmeans.transform(X_embed)
        norm_distances = MinMaxScaler().fit(abs_distances).transform(abs_distances)

        param_arrays = [parameters_to_ndarrays(fr.parameters) for _, fr in results]
        flat_updates = np.stack(
            [np.concatenate([p.ravel() for p in pa]) for pa in param_arrays]
        )
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

        time_start_calc = time.time_ns()
        dists = self._pairwise_sq_dists(flat_updates)
        m = n - f - 2  # number of nearest neighbours to sum
        krum_scores = np.array([np.partition(dists[i], m)[:m].sum() for i in range(n)])
        candidate_idx = np.argpartition(krum_scores, C)[:C]
        candidates = flat_updates[candidate_idx]

        sorted_idx = np.argsort(candidates, axis=0)
        kept_slice = slice(f, C - f)
        trimmed = candidates[sorted_idx[kept_slice, np.arange(dim)], np.arange(dim)]
        bulyan_vector = trimmed.mean(axis=0)
        time_end_calc = time.time_ns()

        agg_list, cursor = [], 0
        for arr in param_arrays[0]:
            num = arr.size
            agg_list.append(
                bulyan_vector[cursor : cursor + num]
                .reshape(arr.shape)
                .astype(arr.dtype)
            )
            cursor += num
        aggregated_parameters = ndarrays_to_parameters(agg_list)

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

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple[ClientProxy, fl.common.FitIns]]:
        """Configure client selection for the next training round.

        During warmup rounds (before begin_removing_from_round), all clients
        participate. After warmup, removes f clients with highest deviation
        scores each round.

        Args:
            server_round: Current round number from the Flower server.
            parameters: Current global model parameters to distribute.
            client_manager: Flower client manager for accessing clients.

        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients.
        """
        available_clients = client_manager.all()

        # Warmup phase: include all clients
        if (
            self.begin_removing_from_round is not None
            and self.current_round <= self.begin_removing_from_round
        ):
            fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
            return [(c, fit_ins) for c in available_clients.values()]

        client_scores = {
            cid: self.client_scores.get(cid, 0.0) for cid in available_clients.keys()
        }
        self.removed_client_ids = set()

        if self.remove_clients:
            n = len(client_scores)
            f = max(0, n - self.num_krum_selections) // 2
            for _ in range(f):
                if not client_scores:
                    break
                eligible = {
                    cid: s
                    for cid, s in client_scores.items()
                    if cid not in self.removed_client_ids
                }
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

        ordered_cids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [
            (available_clients[cid], fit_ins)
            for cid in ordered_cids
            if cid in available_clients
        ]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate client evaluation results and record metrics.

        Records per-client accuracy and loss to strategy_history. Computes
        weighted average loss from non-removed clients only.

        Args:
            server_round: Current round number from the Flower server.
            results: List of (ClientProxy, EvaluateRes) tuples from clients.
            failures: List of failed evaluation results or exceptions.

        Returns:
            Tuple of (aggregated loss, metrics dict).
        """
        self.logger.info(
            "\n" + "-" * 50 + f"AGGREGATION ROUND {server_round}" + "-" * 50
        )

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
        self.strategy_history.insert_round_history_entry(
            loss_aggregated=loss_aggregated
        )

        for cp, ev in results:
            logging.debug(f"Client ID: {cp.cid} Metrics: {ev.metrics} Loss: {ev.loss}")

        self.logger.info(
            f"Round: {server_round} Number of aggregated clients: {num_clients_loss} "
            f"Aggregated loss: {loss_aggregated}"
        )
        return loss_aggregated, {}
