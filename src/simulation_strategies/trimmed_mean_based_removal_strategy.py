import numpy as np
import flwr as fl
import logging

from typing import Dict, List, Optional, Set, Tuple, Union

from flwr.common import Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class TrimmedMeanBasedRemovalStrategy(FedAvg):
    """Trimmed mean aggregation strategy with client removal.

    Computes coordinate-wise trimmed mean to exclude extreme parameter values,
    tracking trim frequency as removal criterion.
    """

    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        trim_ratio: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.trim_ratio = trim_ratio
        self.current_round = 0
        self.client_scores = {}

        self.strategy_history = strategy_history

    def aggregate_fit(
        self, server_round: int, results: List[Tuple], failures: List[BaseException]
    ) -> Tuple[Optional[Union[ndarrays_to_parameters, bytes]], Dict[str, Scalar]]:
        """Aggregate client updates using coordinate-wise trimmed mean.

        Trims the top and bottom trim_ratio values for each parameter coordinate
        and averages the remaining values. Tracks trim frequency as removal
        criterion.

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

        if not results:
            return None, {}

        participating_clients = [client.cid for client, _ in results]
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                client.cid,
            )
            for client, fit_res in results
        ]

        num_clients = len(weights_results)
        num_trim = int(self.trim_ratio * num_clients)

        if num_trim == 0:
            aggregated_weights = self._average_weights(
                [w for w, _, _ in weights_results]
            )

            for cid in participating_clients:
                self.client_scores[cid] = 0.0
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    removal_criterion=0.0,
                )

            self.strategy_history.update_client_participation(
                current_round=self.current_round, removed_client_ids=set()
            )
            return ndarrays_to_parameters(aggregated_weights), {}

        weights_by_layer = list(zip(*[w for w, _, _ in weights_results]))
        aggregated = []
        trimmed_clients: Set[str] = set()
        client_trim_counts = {cid: 0 for _, _, cid in weights_results}
        total_parameters = 0

        for layer_weights in weights_by_layer:
            stacked = np.stack(layer_weights)
            trimmed_layer = []
            num_params_in_layer = (
                np.prod(stacked.shape[1:]) if len(stacked.shape) > 1 else 1
            )
            total_parameters += num_params_in_layer

            for i in range(num_params_in_layer):
                values = (
                    stacked
                    if len(stacked.shape) == 1
                    else stacked.reshape((num_clients, -1))[:, i]
                )
                sorted_indices = np.argsort(values)
                trimmed_indices = sorted_indices[num_trim:-num_trim]
                trimmed_values = values[trimmed_indices]
                trimmed_layer.append(np.mean(trimmed_values))

                removed_this_dim = set(
                    weights_results[j][2] for j in sorted_indices[:num_trim]
                ).union(weights_results[j][2] for j in sorted_indices[-num_trim:])
                trimmed_clients.update(removed_this_dim)

                for cid in removed_this_dim:
                    client_trim_counts[cid] += 1

            aggregated.append(np.array(trimmed_layer).reshape(stacked.shape[1:]))

        for cid in participating_clients:
            trim_frequency = (
                client_trim_counts[cid] / total_parameters
                if total_parameters > 0
                else 0.0
            )
            self.client_scores[cid] = trim_frequency
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=float(trim_frequency),
            )

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=set()
        )

        logging.info(f"clients with trimmed parameters this round: {trimmed_clients}")

        return ndarrays_to_parameters(aggregated), {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure client selection for the next training round.

        During warmup, all clients participate. After warmup, removes the
        client with highest trim frequency if remove_clients is enabled.

        Args:
            server_round: Current round number from the Flower server.
            parameters: Current global model parameters to distribute.
            client_manager: Flower client manager for accessing clients.

        Returns:
            List of (ClientProxy, FitIns) tuples for selected clients.
        """
        currently_removed_client_ids = set()
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

        if self.remove_clients:
            client_id = max(client_scores, key=client_scores.get)
            currently_removed_client_ids.add(client_id)

        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})

        return [
            (available_clients[cid], fit_ins)
            for cid in selected_client_ids
            if cid in available_clients
        ]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results and record metrics.

        Records per-client accuracy and loss to strategy_history. Computes
        weighted average loss from all participating clients.

        Args:
            server_round: Current round number from the Flower server.
            results: List of (ClientProxy, EvaluateRes) tuples from clients.
            failures: List of failed evaluation results or exceptions.

        Returns:
            Tuple of (aggregated loss, metrics dict).
        """
        logging.info("\n" + "-" * 50 + f"AGGREGATION ROUND {server_round}" + "-" * 50)

        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics

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
            client_id = client_metadata.cid

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_id),
                current_round=self.current_round,
                loss=evaluate_res.loss,
            )

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

        logging.info(
            f"Round: {server_round} "
            f"Number of aggregated clients: {number_of_clients_in_loss_calc} "
            f"Aggregated loss: {loss_aggregated} "
        )

        return loss_aggregated, metrics_aggregated

    def _average_weights(self, weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Compute average weights."""
        avg_weights = []
        for layers in zip(*weights):
            stacked = np.stack(layers, axis=0)
            avg_weights.append(np.mean(stacked, axis=0))
        return avg_weights
