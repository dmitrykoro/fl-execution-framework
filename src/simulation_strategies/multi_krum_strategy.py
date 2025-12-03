import time
import numpy as np
import flwr as fl
import torch
import logging
import os

from typing import Optional, Union

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy

from src.output_handlers.directory_handler import DirectoryHandler

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class MultiKrumStrategy(fl.server.strategy.FedAvg):
    """Multi-Krum Byzantine-resilient aggregation strategy.

    Selects the top num_krum_selections clients with lowest Krum scores
    for aggregation, filtering potentially malicious updates.
    """

    def __init__(
        self,
        remove_clients: bool,
        num_of_malicious_clients: int,
        num_krum_selections: int,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.client_scores = {}
        self.removed_client_ids = set()
        self.remove_clients = remove_clients
        self.num_of_malicious_clients = num_of_malicious_clients
        self.num_krum_selections = num_krum_selections
        self.begin_removing_from_round = begin_removing_from_round
        self.current_round = 0
        self.logger = logging.getLogger(f"multi_krum_{id(self)}")
        self.logger.setLevel(logging.INFO)
        out_dir = DirectoryHandler.dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
        self.strategy_history = strategy_history

    def _calculate_chunked_distance(
        self, params1: np.ndarray, params2: np.ndarray, chunk_size: int = 10_000_000
    ) -> float:
        """Calculate L2 distance using chunked processing for large models."""
        total_params = len(params1)
        squared_diff_sum = 0.0

        # Process in chunks to avoid memory overflow
        for start_idx in range(0, total_params, chunk_size):
            end_idx = min(start_idx + chunk_size, total_params)
            chunk1 = params1[start_idx:end_idx]
            chunk2 = params2[start_idx:end_idx]

            # Compute squared difference for this chunk
            diff = chunk1 - chunk2
            squared_diff_sum += np.sum(diff**2)

            # Free memory immediately
            del diff, chunk1, chunk2

        # Return L2 norm (square root of sum of squared differences)
        return np.sqrt(squared_diff_sum)

    def _calculate_multi_krum_scores(
        self, results: list[tuple[ClientProxy, FitRes]], distances: list[float]
    ) -> list[float]:
        """Calculate Multi-Krum scores based on pairwise parameter distances."""
        param_data = [
            fl.common.parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        flat_param_data = [
            np.concatenate([p.flatten() for p in params]) for params in param_data
        ]
        param_data = flat_param_data
        num_clients = len(param_data)
        param_size = len(param_data[0]) if param_data else 0
        use_chunked = param_size > 50_000_000

        if use_chunked:
            logging.info(
                f"Multi-Krum using chunked distance calculation for large model "
                f"({param_size:,} parameters)"
            )

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if use_chunked:
                    distances[i, j] = self._calculate_chunked_distance(
                        param_data[i], param_data[j]
                    )
                else:
                    distances[i, j] = np.linalg.norm(param_data[i] - param_data[j])
                distances[j, i] = distances[i, j]

        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[: self.num_krum_selections - 2])
            scores.append(score)

        return scores

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate client model updates using the Multi-Krum algorithm.

        Computes Multi-Krum scores for each client based on pairwise parameter
        distances, then selects the top num_krum_selections clients with lowest
        scores for aggregation.

        Args:
            server_round: Current round number from the Flower server.
            results: List of (ClientProxy, FitRes) tuples from clients.
            failures: List of failed client results or exceptions.

        Returns:
            Tuple of (aggregated parameters, metrics dict).
        """
        self.current_round += 1

        if self.strategy_history:
            self.strategy_history.update_client_malicious_status(server_round)

        clustering_param_data = []
        for client_proxy, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_tensor_list = [torch.Tensor(arr) for arr in client_params]
            flattened_param_list = [param.flatten() for param in params_tensor_list]
            param_tensor = torch.cat(flattened_param_list)
            clustering_param_data.append(param_tensor)

        X = np.array(clustering_param_data)
        kmeans = KMeans(n_clusters=1, init="k-means++").fit(X)
        distances = kmeans.transform(X)

        scaler = MinMaxScaler()
        scaler.fit(distances)
        normalized_distances = scaler.transform(distances)

        distances = np.zeros((len(results), len(results)))
        time_start_calc = time.time_ns()

        multi_krum_scores = self._calculate_multi_krum_scores(results, distances)

        selected_indices = np.argsort(multi_krum_scores)[: self.num_krum_selections]
        selected_clients = [results[i] for i in selected_indices]
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, selected_clients, failures
        )

        time_end_calc = time.time_ns()

        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end_calc - time_start_calc
        )

        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            score = float(multi_krum_scores[i])
            self.client_scores[client_id] = score

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=float(score),
                absolute_distance=float(distances[i][0]),
            )

            self.logger.info(
                f"Aggregation round: {server_round} "
                f"Client ID: {client_id} "
                f"Multi-Krum Score: {score} "
                f"Normalized Distance: {normalized_distances[i][0]}"
            )

        return aggregated_parameters, aggregated_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, fl.common.FitIns]]:
        """Configure client selection for the next training round.

        During warmup rounds (before begin_removing_from_round), all clients
        participate. After warmup, removes clients with highest Multi-Krum
        scores until only num_krum_selections clients remain.

        Args:
            server_round: Current round number from the Flower server.
            parameters: Current global model parameters to distribute.
            client_manager: Flower client manager for accessing clients.

        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients.
        """
        available_clients = client_manager.all()

        if (
            self.begin_removing_from_round is not None
            and self.current_round <= self.begin_removing_from_round
        ):
            fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
            return [(client, fit_ins) for client in available_clients.values()]

        client_scores = {
            client_id: self.client_scores.get(client_id, 0)
            for client_id in available_clients.keys()
        }
        self.removed_client_ids = set()

        while (
            len(self.removed_client_ids)
            < len(self.client_scores) - self.num_krum_selections
        ):
            eligible_clients = {
                cid: score
                for cid, score in client_scores.items()
                if cid not in self.removed_client_ids
            }
            if eligible_clients:
                client_id_to_remove = max(eligible_clients, key=eligible_clients.get)
                self.removed_client_ids.add(client_id_to_remove)

        self.logger.info(
            f"Removed clients at round {self.current_round} are : {self.removed_client_ids}"
        )
        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )

        return [
            (available_clients[cid], fit_ins)
            for cid in selected_client_ids
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

        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics
            accuracy_matrix["cid"] = cid

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=accuracy_matrix["accuracy"],
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
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(
            loss_aggregated=loss_aggregated
        )

        for result in results:
            logging.debug(f"Client ID: {result[0].cid}")
            logging.debug(f"Metrics: {result[1].metrics}")
            logging.debug(f"Loss: {result[1].loss}")

        metrics_aggregated = {}

        self.logger.info(
            f"Round: {server_round} "
            f"Number of aggregated clients: {number_of_clients_in_loss_calc} "
            f"Aggregated loss: {loss_aggregated}"
        )

        return loss_aggregated, metrics_aggregated
