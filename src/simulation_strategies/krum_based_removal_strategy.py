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
from flwr.server.strategy.krum import Krum

class KrumBasedRemovalStrategy(Krum):
    def __init__(self, remove_clients: bool, num_malicious_clients: int, num_krum_selections: int, begin_removing_from_round: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_scores = {}
        self.removed_client_ids = set() 
        self.remove_clients = remove_clients
        self.num_malicious_clients = num_malicious_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.current_round = 0
        self.rounds_history = {}
        self.num_krum_selections = num_krum_selections

    def _calculate_krum_scores(self, results: List[Tuple[ClientProxy, FitRes]], distances: List[float]) -> List[float]:
        """
        Calculate Krum scores based on the parameter differences between clients.

        Args:
            results (List[Tuple[ClientProxy, FitRes]]): List of client proxies and their fit results.
            distances: empty array to store distances

        Returns:
            List[float]: Krum scores for each client.
        """
        param_data = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        flat_param_data = [np.concatenate([p.flatten() for p in params]) for params in param_data]
        param_data = flat_param_data
        num_clients = len(param_data)

        # Compute pairwise distances between clients' model updates
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distances[i, j] = np.linalg.norm(param_data[i] - param_data[j])
                distances[j, i] = distances[i, j]

        # Calculate Krum scores based on the distances
        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:num_clients - self.num_malicious_clients - 2])
            scores.append(score)
        return scores
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1
        self.rounds_history[f'{self.current_round}'] = {}

        # clustering
        clustering_param_data = []
        for client_proxy, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_tensor_list = [torch.Tensor(arr) for arr in client_params]
            flattened_param_list = [param.flatten() for param in params_tensor_list]
            param_tensor = torch.cat(flattened_param_list)
            # extract mean of weights and bias of the last layer (fc3)
            clustering_param_data.append(param_tensor)
            
        # perform clustering
        X = np.array(clustering_param_data)
        kmeans = KMeans(n_clusters=1, init='k-means++').fit(X)
        distances = kmeans.transform(X)

        scaler = MinMaxScaler()
        scaler.fit(distances)
        normalized_distances = scaler.transform(distances)

        distances = np.zeros((len(results), len(results)))

        time_start_calc = time.time_ns()

        krum_scores = self._calculate_krum_scores(results, distances)
        time_end_calc = time.time_ns()
        self.rounds_history[f'{self.current_round}']['round_info'] = {}
        self.rounds_history[f'{self.current_round}']['round_info']['score_calculation_time_nanos'] = time_end_calc - time_start_calc

        self.rounds_history[f'{self.current_round}']['client_info'] = {}

        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            score = float(krum_scores[i])
            self.client_scores[client_id] = score
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}'] = {}
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['removal_criterion'] = score
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['absolute_distance'] = float(distances[i][0])
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['normalized_distance'] = float(normalized_distances[i][0])

            if self.current_round == 1:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = False
            else:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = (
                    self.rounds_history)[f'{self.current_round - 1}']['client_info'][f'client_{client_id}']['is_removed']

            logging.info(f'Aggregation round: {server_round} Client ID: {client_id} Krum Score: {score} Normalized Distance: {normalized_distances[i][0]}')
        
        # Select the client with the minimum Krum score
        min_krum_score_index = np.argmin(krum_scores)
        min_krum_client = results[min_krum_score_index]

        # Log the selected client for aggregation
        selected_client_id = min_krum_client[0].cid
        logging.info(f"Selected client for aggregation: {selected_client_id} with Krum Score: {krum_scores[min_krum_score_index]}")


        # Aggregate only the selected client's parameters
        selected_clients = [min_krum_client]  # Aggregate only the client with the minimum Krum score
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, selected_clients, failures)


        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        # fetch the available clients as a dictionary
        available_clients = client_manager.all() # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # in the warmup rounds, select all clients
        if self.current_round <= self.begin_removing_from_round:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        # fetch the krum based scores for all available clients
        client_scores = {client_id: self.client_scores.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # in the first round after warmup, remove the client with the highest Krum score
            client_id = max(client_scores, key=client_scores.get)
            logging.info(f"Removing client with highest Krum score: {client_id}")
            self.removed_client_ids.add(client_id)
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = True
        
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
            accuracy_matrix['cid'] = cid
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{cid}']['accuracy'] = accuracy_matrix['accuracy']

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            # update history
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_metadata.cid}']['loss'] = evaluate_res.loss

            if client_metadata.cid not in self.removed_client_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)

        for result in results:
            logging.debug(f'Client ID: {result[0].cid}')
            logging.debug(f'Metrics: {result[1].metrics}')
            logging.debug(f'Loss: {result[1].loss}')

        metrics_aggregated = {}

        logging.info(f'Round: {server_round} Number of aggregated clients: {number_of_clients_in_loss_calc} Aggregated loss: {loss_aggregated}')
        return loss_aggregated, metrics_aggregated
