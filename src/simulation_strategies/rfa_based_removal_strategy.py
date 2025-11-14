import time
import numpy as np
import flwr as fl
import torch
import logging

from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

class RFABasedRemovalStrategy(FedAvg):
    def __init__(
            self,
            remove_clients: bool,
            begin_removing_from_round: int,
            weighted_median_factor: float = 1.0,
            *args, **kwargs
    ):
        self.strategy_history = kwargs.pop('strategy_history', None)
        super().__init__(*args, **kwargs)

        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.weighted_median_factor = weighted_median_factor
        self.current_round = 0
        self.removed_client_ids = set()
        self.rounds_history = {}
        self.client_scores = {}

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        # Increment the current round counter.
        self.current_round += 1
        self.rounds_history[f'{self.current_round}'] = {}
        aggregate_clients = []

        # Filter clients that have not been removed.
        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_client_ids:
                aggregate_clients.append(result)
        
        if not aggregate_clients:
            return super().aggregate_fit(server_round, results, failures)

        # Convert parameters to numpy arrays for aggregation.
        param_data = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in aggregate_clients]
        stacked_params = np.stack([np.concatenate([p.flatten() for p in params]) for params in param_data])

        
        time_start_calc = time.time_ns()
        # Calculate the geometric median (RFA) and use it as the removal criterion.
        geometric_median = self._geometric_median(stacked_params)
        weighted_geometric_median = geometric_median * self.weighted_median_factor
        time_end_calc = time.time_ns()

        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end_calc - time_start_calc
        )

        # Perform clustering for monitoring and logging purposes.
        clustering_param_data = []
        for client_proxy, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_tensor_list = [torch.Tensor(arr) for arr in client_params]
            flattened_param_list = [param.flatten() for param in params_tensor_list]
            param_tensor = torch.cat(flattened_param_list)
            clustering_param_data.append(param_tensor)

        # Perform clustering using KMeans.
        X = np.array(clustering_param_data)
        kmeans = KMeans(n_clusters=1, init='k-means++').fit(X)
        distances = kmeans.transform(X)

        scaler = MinMaxScaler()
        scaler.fit(distances)
        normalized_distances = scaler.transform(distances)

        # Store the removal criterion (deviation from weighted geometric median) and other info for each client.
        for i, (client_proxy, _) in enumerate(aggregate_clients):
            client_id = client_proxy.cid
            deviation = np.linalg.norm(stacked_params[i] - weighted_geometric_median)
            self.client_scores[client_id] = deviation

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=deviation
            )

            logging.info(f'Aggregation round: {server_round} Client ID: {client_id} Deviation: {deviation} Normalized Distance: {normalized_distances[i][0]}')

        # Aggregate the parameters using the weighted geometric median.
        aggregated_parameters_list = []
        start_idx = 0
        for param in param_data[0]:
            param_size = param.size
            aggregated_param = np.reshape(weighted_geometric_median[start_idx:start_idx + param_size], param.shape)
            aggregated_parameters_list.append(aggregated_param)
            start_idx += param_size
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_parameters_list)
        aggregated_metrics = {}

        return aggregated_parameters, aggregated_metrics

    def _geometric_median(self, points: np.ndarray, max_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
        """
        Calculate the geometric median of a set of points.

        Args:
            points (np.ndarray): Numpy array of points for which the geometric median is to be calculated.
            max_iter (int): Maximum number of iterations for convergence.
            tol (float): Tolerance for convergence.

        Returns:
            np.ndarray: The calculated geometric median.
        """
        median = np.mean(points, axis=0)
        for _ in range(max_iter):
            distances = np.linalg.norm(points - median, axis=1)
            non_zero_distances = distances[distances != 0]
            if len(non_zero_distances) == 0:
                return median
            weights = 1 / np.maximum(distances, tol)
            new_median = np.average(points, axis=0, weights=weights)
            if np.linalg.norm(new_median - median) < tol:
                return new_median
            median = new_median
        return median

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        # Fetch available clients as a dictionary.
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # Select all clients in the warmup rounds.
        if self.current_round <= self.begin_removing_from_round:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        # Select clients that have not been removed in previous rounds.
        client_scores = {client_id: self.client_scores.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # Remove clients with the highest deviation if applicable.
            client_id = max(client_scores, key=client_scores.get)
            logging.info(f"Removing client with highest deviation: {client_id}")

            self.removed_client_ids.add(client_id)

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )
        
        logging.info(f"removed clients are : {self.removed_client_ids}")

        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {})
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

            acc = accuracy_matrix.get('accuracy')

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=acc,
            )

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            client_id = client_metadata.cid

            if client_id not in self.removed_client_ids:
                self.strategy_history.insert_single_client_history_entry(
                    client_id=int(client_id), current_round=self.current_round, loss=evaluate_res.loss
                )

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
