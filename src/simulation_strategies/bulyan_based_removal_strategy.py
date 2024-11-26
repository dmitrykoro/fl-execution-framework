import time
import numpy as np
import flwr as fl
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from flwr.common import FitRes, Parameters, Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.bulyan import Bulyan
from flwr.server.strategy.aggregate import weighted_loss_avg

class BulyanBasedRemovalStrategy(Bulyan):
    def __init__(self, remove_clients: bool, num_malicious_clients: int, begin_removing_from_round: int, *args, **kwargs):
        super().__init__(num_malicious_clients=num_malicious_clients, *args, **kwargs)
        self.remove_clients = remove_clients
        self.num_malicious_clients = num_malicious_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.current_round = 0
        self.removed_client_ids = set()
        self.rounds_history = {}
        self.client_scores = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Increment the current round counter.
        self.current_round += 1
        self.rounds_history[f'{self.current_round}'] = {'client_info': {}}
        aggregate_clients = [
            result for result in results if result[0].cid not in self.removed_client_ids
        ]

        if len(aggregate_clients) == 0:
            logging.warning("No clients available for aggregation after filtering removed clients.")
            return None, {}

        time_start_calc = time.time_ns()
        # Calculate removal criteria scores for each client
        scores = self._calculate_removal_criteria(aggregate_clients)

        # Select the top clients based on scores
        num_clients = len(aggregate_clients)
        selected_indices = np.argsort(scores)[:max(1, num_clients - self.num_malicious_clients)]
        selected_clients = [aggregate_clients[i] for i in selected_indices]

        # Aggregate the selected parameters
        param_data = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in selected_clients]
        stacked_params = np.stack([np.concatenate([p.flatten() for p in params]) for params in param_data])
        aggregated_parameters = np.mean(stacked_params, axis=0)
        aggregated_parameters_list = []
        start_idx = 0
        for param in param_data[0]:
            param_size = param.size
            aggregated_param = np.reshape(aggregated_parameters[start_idx:start_idx + param_size], param.shape)
            aggregated_parameters_list.append(aggregated_param)
            start_idx += param_size
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters_list)

        time_end_calc = time.time_ns()
        self.rounds_history[f'{self.current_round}']['round_info'] = {}
        self.rounds_history[f'{self.current_round}']['round_info']['score_calculation_time_nanos'] = time_end_calc - time_start_calc

        # Update round history and logging
        for i, (client_proxy, _) in enumerate(aggregate_clients):
            client_id = client_proxy.cid
            deviation = scores[i]
            self.client_scores[client_id] = deviation
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}'] = {
                'removal_criterion': deviation,
                'is_removed': self.rounds_history.get(f'{self.current_round - 1}', {}).get('client_info', {}).get(f'client_{client_id}', {}).get('is_removed', False),
                'absolute_distance': deviation,
                'normalized_distance': deviation  # Added normalized distance for logging
            }
            logging.info(f'Aggregation round: {server_round} Client ID: {client_id} Score: {deviation}')

        return aggregated_parameters, {}

    def _calculate_removal_criteria(self, results: List[Tuple[ClientProxy, FitRes]]) -> List[float]:
        """
        Calculate removal criteria scores for each client based on pairwise distances.

        Args:
            results (List[Tuple[ClientProxy, FitRes]]): List of client proxies and their fit results.

        Returns:
            List[float]: Scores for each client based on the removal criteria.
        """
        param_data = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        stacked_params = np.stack([np.concatenate([p.flatten() for p in params]) for params in param_data])
        num_clients = len(stacked_params)
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distances[i, j] = np.linalg.norm(stacked_params[i] - stacked_params[j])
                distances[j, i] = distances[i, j]

        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:num_clients - self.num_malicious_clients - 2])
            scores.append(score)
        return scores

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        available_clients = client_manager.all()
        fit_ins = fl.common.FitIns(parameters, {})

        # Remove clients with the highest score if removal is enabled and past the warmup rounds
        if self.remove_clients and self.current_round > self.begin_removing_from_round:
            client_to_remove = max(self.client_scores, key=self.client_scores.get)
            logging.info(f"Removing client with highest score: {client_to_remove}")
            self.removed_client_ids.add(client_to_remove)
            if f'client_{client_to_remove}' not in self.rounds_history[f'{self.current_round}']['client_info']:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_to_remove}'] = {}
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_to_remove}']['is_removed'] = True

        selected_clients = [
            (client, fit_ins) for cid, client in available_clients.items()
            if cid not in self.removed_client_ids
        ]
        logging.info(f"configure_fit: Selected clients {[client[0].cid for client in selected_clients]}")
        return selected_clients

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        logging.info('\n' + '-' * 50 + f'AGGREGATION ROUND {server_round}' + '-' * 50)

        # Ensure round history is consistent
        previous_round = str(self.current_round - 1)
        if previous_round in self.rounds_history:
            for client_id in self.rounds_history[previous_round]['client_info'].keys():
                if client_id not in self.rounds_history[f'{self.current_round}']['client_info']:
                    self.rounds_history[f'{self.current_round}']['client_info'][client_id] = self.rounds_history[previous_round]['client_info'][client_id].copy()
                    if client_id in self.removed_client_ids:
                        self.rounds_history[f'{self.current_round}']['client_info'][client_id]['accuracy'] = None
                        self.rounds_history[f'{self.current_round}']['client_info'][client_id]['loss'] = None

        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics

            if f'client_{cid}' not in self.rounds_history[f'{self.current_round}']['client_info']:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{cid}'] = {}

            if cid not in self.removed_client_ids:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{cid}']['accuracy'] = accuracy_matrix.get('accuracy', None)

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            client_id = client_metadata.cid

            if client_id not in self.removed_client_ids:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['loss'] = evaluate_res.loss
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)

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
