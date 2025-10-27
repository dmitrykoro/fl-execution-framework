import time
import numpy as np
import flwr as fl
import logging

from typing import Optional, Union

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class TrimmedMeanBasedRemovalStrategy(FedAvg):
    def __init__(
            self,
            remove_clients: bool,
            begin_removing_from_round: int,
            strategy_history: SimulationStrategyHistory,
            trim_ratio: float = 0.1,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.trim_ratio = trim_ratio
        self.current_round = 0
        self.removed_client_ids = set()
        self.client_scores = {}

        self.strategy_history = strategy_history

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple],
            failures: list[BaseException]
    ) -> tuple[Optional[Union[ndarrays_to_parameters, bytes]], dict[str, Scalar]]:

        self.current_round += 1

        # Update client.is_malicious based on attack_schedule for dynamic attacks
        if self.strategy_history:
            self.strategy_history.update_client_malicious_status(server_round)

        if not results:
            return None, {}

        # Track all clients that submitted updates
        participating_clients = [client.cid for client, _ in results]

        # Extract weights and client IDs
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client.cid)
                           for client, fit_res in results]

        num_clients = len(weights_results)
        num_trim = int(self.trim_ratio * num_clients)

        if num_trim == 0:
            # No trimming needed
            aggregated_weights = self._average_weights([w for w, _, _ in weights_results])
            self.strategy_history.update_client_participation(
                current_round=self.current_round,
                removed_client_ids=set()
            )
            return ndarrays_to_parameters(aggregated_weights), {}

        # Transpose weights for each layer
        weights_by_layer = list(zip(*[w for w, _, _ in weights_results]))
        aggregated = []

        trimmed_clients: set[str] = set()

        for layer_weights in weights_by_layer:
            stacked = np.stack(layer_weights)  # Shape: (n_clients, layer_shape...)

            # Flatten weights across clients
            trimmed_layer = []
            for i in range(np.prod(stacked.shape[1:]) if len(stacked.shape) > 1 else 1):
                # For each scalar value in the layer (if multidimensional)
                values = stacked if len(stacked.shape) == 1 else stacked.reshape((num_clients, -1))[:, i]
                sorted_indices = np.argsort(values)
                trimmed_indices = sorted_indices[num_trim:-num_trim]
                trimmed_values = values[trimmed_indices]
                trimmed_layer.append(np.mean(trimmed_values))

                # Track which clients were trimmed
                removed_this_dim = set(
                    weights_results[j][2] for j in sorted_indices[:num_trim]
                ).union(
                    weights_results[j][2] for j in sorted_indices[-num_trim:]
                )
                trimmed_clients.update(removed_this_dim)

            aggregated.append(np.array(trimmed_layer).reshape(stacked.shape[1:]))

        # Log trimmed clients for this round
        self.strategy_history.update_client_participation(
            current_round=self.current_round,
            removed_client_ids=removed_this_dim
        )

        logging.info(f"removed clients are : {removed_this_dim}")

        return ndarrays_to_parameters(aggregated), {}

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters, client_manager
    ) -> list[tuple[ClientProxy, fl.common.FitIns]]:

        currently_removed_client_ids = set()

        # Fetch available clients as a dictionary.
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # Select all clients in the warmup rounds.
        if self.current_round <= self.begin_removing_from_round:
            fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
            return [(client, fit_ins) for client in available_clients.values()]

        # Select clients that have not been removed in previous rounds.
        client_scores = {client_id: self.client_scores.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # Remove clients with the highest score if applicable.
            client_id = max(client_scores, key=client_scores.get)
            currently_removed_client_ids.add(client_id)

        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})

        return [(available_clients[cid], fit_ins) for cid in selected_client_ids if cid in available_clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> tuple[Optional[float], dict[str, Scalar]]:

        logging.info('\n' + '-' * 50 + f'AGGREGATION ROUND {server_round}' + '-' * 50)

        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics

            if cid not in self.removed_client_ids:
                self.strategy_history.insert_single_client_history_entry(
                    client_id=int(cid),
                    current_round=self.current_round,
                    accuracy=accuracy_matrix['accuracy']
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
                loss=evaluate_res.loss
            )

            if client_id not in self.removed_client_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)

        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        for result in results:
            logging.debug(f'Client ID: {result[0].cid}')
            logging.debug(f'Metrics: {result[1].metrics}')
            logging.debug(f'Loss: {result[1].loss}')

        metrics_aggregated = {}

        logging.info(
            f'Round: {server_round} '
            f'Number of aggregated clients: {number_of_clients_in_loss_calc} '
            f'Aggregated loss: {loss_aggregated} '
        )

        return loss_aggregated, metrics_aggregated

    def _average_weights(self, weights: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Compute average weights."""
        num_weights = len(weights)
        avg_weights = []
        for layers in zip(*weights):
            stacked = np.stack(layers, axis=0)
            avg_weights.append(np.mean(stacked, axis=0))
        return avg_weights