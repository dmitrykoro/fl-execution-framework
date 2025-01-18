import numpy as np
import flwr as fl
import torch
import math as m
import matplotlib.pyplot as plt
import logging
import time

from typing import Dict, List, Optional, Tuple, Union

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy

from data_models.round_info import RoundsInfo
from data_models.client_info import ClientInfo
from data_models.simulation_strategy_history import SimulationStrategyHistory


class PIDBasedRemovalStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            remove_clients: bool,
            pid_threshold: float,
            begin_removing_from_round: int,
            ki: float,
            kd: float,
            kp: float,
            strategy_history: SimulationStrategyHistory,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.client_pids = {}
        self.client_distance_sums = {}
        self.client_distances = {}
        self.current_round = 0
        self.removed_client_ids = set()

        self.remove_clients = remove_clients
        self.pid_threshold = pid_threshold
        self.begin_removing_from_round = begin_removing_from_round

        self.ki = ki
        self.kd = kd
        self.kp = kp

        self.current_threshold = pid_threshold
        self.rounds_history = {}

        self.strategy_history = strategy_history

    def calculate_pid(self, client_id, distance):
        """Calculate pid."""

        p = distance * self.kp

        if self.current_round == 1:
            return p
        else:
            curr_sum = self.client_distance_sums.get(client_id, 0)
            i = curr_sum * self.ki
            prev_distance = self.client_distances.get(client_id, 0)
            d = self.kd * (distance-prev_distance) 
            
            return p + i + d
        
    def calculate_all_pid_scores(self, results, normalized_distances) -> List[float]:
        pid_scores = []
        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            curr_dist = normalized_distances[i][0]

            new_PID = self.calculate_single_client_pid(client_id, curr_dist)
            self.client_pids[client_id] = new_PID
            pid_scores.append(new_PID)
        return pid_scores

    @staticmethod
    def cosine_similarity(tensor1, tensor2):
        """Calculate cosine similarity."""

        dot_product = torch.dot(tensor1, tensor2)
        norm1 = torch.norm(tensor1)
        norm2 = torch.norm(tensor2)
        return dot_product / (norm1 * norm2)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1
        self.rounds_history[f'{self.current_round}'] = {}
        aggregate_clients = []

        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_client_ids:
                aggregate_clients.append(result)

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, aggregate_clients, failures)

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

        self.rounds_history[f'{self.current_round}']['client_info'] = {}

        time_start_calc = time.time_ns()
        pids = self.calculate_all_pid_scores(results, normalized_distances)
        time_end_calc = time.time_ns()
        self.rounds_history[f'{self.current_round}']['round_info'] = {}
        self.rounds_history[f'{self.current_round}']['round_info']['score_calculation_time_nanos'] = time_end_calc - time_start_calc

        self.strategy_history.insert_round_history_entry(score_calculation_time_nanos=time_end_calc - time_start_calc)

        self.rounds_history[f'{self.current_round}']['client_info'] = {}

        counted_pids = []
        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            curr_dist = normalized_distances[i][0]

            new_PID = pids[i]
            self.client_pids[client_id] = new_PID

            if not client_id in self.removed_client_ids:
                counted_pids.append(new_PID)

            self.client_distances[client_id] = curr_dist
            curr_sum = self.client_distance_sums.get(client_id, 0)
            self.client_distance_sums[client_id] = curr_dist + curr_sum

            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}'] = {}
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['removal_criterion'] = float(new_PID)
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['absolute_distance'] = float(distances[i][0])
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['normalized_distance'] = float(normalized_distances[i][0])

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=float(new_PID),
                absolute_distance=float(distances[i][0])
            )

            if self.current_round == 1:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = False
            else:
                self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = (
                    self.rounds_history[f'{self.current_round - 1}']['client_info'][f'client_{client_id}']['is_removed']
                )

            logging.info(
                f'Aggregation round: {server_round} '
                f'Client ID: {client_id} '
                f'PID: {new_PID} '
                f'Normalized Distance: {normalized_distances[i][0]} '
            )

        pid_avg = np.mean(counted_pids)
        pid_std = np.std(counted_pids)
        self.rounds_history[f'{self.current_round}']['average'] = pid_avg
        self.rounds_history[f'{self.current_round}']['standard_deviation'] = pid_std
        self.current_threshold = pid_avg + (2 * pid_std)

        self.strategy_history.insert_round_history_entry(removal_threshold=self.current_threshold)

        logging.debug(self.current_threshold)

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        # fetch the available clients as a dictionary
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # in the warmup rounds, select all clients
        if self.current_round <= self.begin_removing_from_round - 1:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        # fetch the Trust and Reputation scores for all available clients
        client_pids = {client_id: self.client_pids.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # remove clients with PID lower than threshold.
            for client_id, pid in client_pids.items():
                if pid > self.current_threshold and client_id not in self.removed_client_ids:
                    logging.info(f"Removing client with PID greater than Threshold: {client_id}")
                    # add this client to the removed_clients list
                    self.removed_client_ids.add(client_id)
                    self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_id}']['is_removed'] = True
                    self.strategy_history.insert_single_client_history_entry(
                        current_round=self.current_round, client_id=int(client_id), aggregation_participation=0
                    )

                else:
                    self.strategy_history.insert_single_client_history_entry(
                        current_round=self.current_round, client_id=int(client_id), aggregation_participation=1
                    )

        logging.info(f"removed clients are : {self.removed_client_ids}")

        # select clients based on updated PID and available clients
        sorted_client_ids = sorted(client_pids, key=client_pids.get, reverse=True)
        selected_client_ids = sorted_client_ids

        # create training configurations for selected clients
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
            self.rounds_history[f'{self.current_round}']['client_info'][f'client_{client_metadata.cid}']['loss'] = evaluate_res.loss

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_metadata.cid),
                current_round=self.current_round,
                loss=evaluate_res.loss
            )

            if client_metadata.cid not in self.removed_client_ids:
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
