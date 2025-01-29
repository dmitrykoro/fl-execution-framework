import time
import numpy as np
import flwr as fl
import logging

from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from data_models.simulation_strategy_history import SimulationStrategyHistory


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
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Increment the current round counter.
        self.current_round += 1
        aggregate_clients = []

        # Filter clients that have not been removed.
        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_client_ids:
                aggregate_clients.append(result)

        # Convert parameters to numpy arrays for aggregation.
        param_data = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in aggregate_clients]
        if len(param_data) == 0:
            logging.warning("No clients available for aggregation after filtering removed clients.")
            return None, {}

        stacked_params = np.stack([np.concatenate([p.flatten() for p in params]) for params in param_data])

        time_start_calc = time.time_ns()
        # Calculate norms for each client's parameter vector.
        norms = np.linalg.norm(stacked_params, axis=1)
        self._assign_scores(norms, aggregate_clients)

        # Sort clients based on their norms and trim them.
        trimmed_indices = self._trim_clients(norms)

        # Aggregate using the trimmed mean.
        if len(trimmed_indices) == 0:
            logging.warning("No clients left after trimming. Using all available clients for aggregation.")
            trimmed_params = stacked_params
        else:
            trimmed_params = stacked_params[trimmed_indices]
        trimmed_mean = np.mean(trimmed_params, axis=0)

        time_end_calc = time.time_ns()

        self.strategy_history.insert_round_history_entry(score_calculation_time_nanos=time_end_calc - time_start_calc)

        # Store the removal criterion and other info for each client.
        for i, (client_proxy, _) in enumerate(aggregate_clients):
            client_id = client_proxy.cid
            norm = norms[i]
            self.client_scores[client_id] = norm

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=float(norm),
                absolute_distance=float(norm)
            )

            logging.info(f'Aggregation round: {server_round} Client ID: {client_id} Norm: {norm} Normalized Distance: {norm / np.max(norms)}')

        # Aggregate the parameters using the trimmed mean.
        aggregated_parameters_list = []
        start_idx = 0
        for param in param_data[0]:
            param_size = param.size
            aggregated_param = np.reshape(trimmed_mean[start_idx:start_idx + param_size], param.shape)
            aggregated_parameters_list.append(aggregated_param)
            start_idx += param_size
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters_list)
        aggregated_metrics = {}

        return aggregated_parameters, aggregated_metrics

    def _assign_scores(
            self,
            scores: np.ndarray,
            aggregate_clients: List[Tuple[ClientProxy, FitRes]]
    ):
        """
        Assign scores to each client.
        """
        for i, (_, _) in enumerate(aggregate_clients):
            client_id = aggregate_clients[i][0].cid
            self.client_scores[client_id] = scores[i]

    def _trim_clients(self, scores: np.ndarray) -> np.ndarray:
        """
        Trim clients based on their scores.
        """
        sorted_indices = np.argsort(scores)
        num_trim = int(len(sorted_indices) * self.trim_ratio)
        if len(sorted_indices) - 2 * num_trim <= 0:
            logging.warning("Not enough clients to perform trimming. Using all available clients for aggregation.")
            return np.arange(len(sorted_indices))
        return sorted_indices[num_trim:-num_trim]

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:

        # Fetch available clients as a dictionary.
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # Select all clients in the warmup rounds.
        if self.current_round <= self.begin_removing_from_round:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        # Select clients that have not been removed in previous rounds.
        client_scores = {client_id: self.client_scores.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # Remove clients with the highest score if applicable.
            client_id = max(client_scores, key=client_scores.get)
            logging.info(f"Removing client with highest score: {client_id}")
            self.removed_client_ids.add(client_id)

        logging.info(f"removed clients are : {self.removed_client_ids}")

        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {})

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )

        return [(available_clients[cid], fit_ins) for cid in selected_client_ids if cid in available_clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

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
