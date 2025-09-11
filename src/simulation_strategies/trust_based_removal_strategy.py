import numpy as np
import flwr as fl
import torch
import math as m
import logging

from typing import Dict, List, Optional, Tuple, Union

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class TrustBasedRemovalStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            remove_clients: bool,
            beta_value: float,
            trust_threshold: float,
            begin_removing_from_round: int,
            strategy_history: SimulationStrategyHistory,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.client_reputations = {}
        self.current_round = 0
        self.client_trusts = {}
        self.removed_client_ids = set()

        self.remove_clients = remove_clients
        self.beta_value = beta_value
        self.trust_threshold = trust_threshold
        self.begin_removing_from_round = begin_removing_from_round

        self.strategy_history = strategy_history

    def calculate_reputation(self, client_id, truth_value):
        """Calculate initial reputation."""

        if self.current_round == 1:
            return truth_value
        else:
            prev_reputation = self.client_reputations.get(client_id, 0)
            return self.update_reputation(prev_reputation, truth_value, self.current_round)

    def update_reputation(self, prev_reputation, truth_value, current_round):
        """Update reputation."""

        if truth_value >= 0.5:
            updated_reputation = (prev_reputation + truth_value) - (prev_reputation / current_round)
        else:
            temp = -(1 - (truth_value * (prev_reputation / current_round)))
            updated_reputation = (prev_reputation + truth_value) - np.exp(temp)
        updated_reputation = self.beta_value * updated_reputation + (1 - self.beta_value) * prev_reputation

        if updated_reputation > 1.0:
            return 1.0
        elif updated_reputation < 0.0:
            return 0.0

        return updated_reputation[0]

    def calculate_trust(self, client_id, reputation, d):
        """Function to get previous rounds trust value and calculate trust."""

        if self.current_round == 1:
            prev_trust = 0
        else:
            prev_trust = self.client_trusts.get(client_id, 0)
        return self.update_trust(prev_trust, reputation, d)

    def update_trust(self, prev_trust, reputation, d):
        """Calculates trust based on reputation value of a client."""
        # Convert numpy arrays to scalars if needed
        if hasattr(d, 'item'):
            d = d.item()
        if hasattr(reputation, 'item'):
            reputation = reputation.item()
        if hasattr(prev_trust, 'item'):
            prev_trust = prev_trust.item()

        trust = m.sqrt(m.pow(reputation, 2) + m.pow(d, 2)) - m.sqrt(m.pow(1 - reputation, 2) + m.pow(1 - d, 2))
        trust = self.beta_value * trust + (1 - self.beta_value) * prev_trust

        if trust > 1.0:
            trust = 1.0
        if trust < 0.0:
            trust = 0.0
        return trust

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        self.current_round += 1
        aggregate_clients = []

        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_client_ids:
                aggregate_clients.append(result)

        if not aggregate_clients:
            return super().aggregate_fit(server_round, results, failures)

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

        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            truth_value = 1 - normalized_distances[i]

            new_reputation = self.calculate_reputation(client_id, truth_value)
            new_trust = self.calculate_trust(client_id, new_reputation, truth_value)

            self.client_reputations[client_id] = new_reputation
            self.client_trusts[client_id] = new_trust

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=float(new_trust.item()) if hasattr(new_trust, 'item') else float(new_trust),
                absolute_distance=float(distances[i][0].item()) if hasattr(distances[i][0], 'item') else float(distances[i][0])
            )

            logging.info(
                f'Aggregation round: {server_round} '
                f'Client ID: {client_id} '
                f'Reputation: {new_reputation} '
                f'Trust: {new_trust} '
                f'Normalized Distance: {normalized_distances[i][0]} '
            )

        self.strategy_history.insert_round_history_entry(removal_threshold=self.trust_threshold)

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        # fetch the available clients as a dictionary
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # in the warmup rounds, select all clients
        if self.current_round <= self.begin_removing_from_round - 1:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        # fetch the Trust and Reputation scores for all available clients
        client_trusts = {client_id: self.client_trusts.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # in the first round after warmup, remove the client with the lowest TRUST
            if self.current_round == self.begin_removing_from_round:
                client_id = min(client_trusts, key=client_trusts.get)
                logging.info(f"Removing client with lowest TRUST: {client_id}")
                # add this client to the removed_clients list
                self.removed_client_ids.add(client_id)
                # add to history
            else:
                # remove clients with trust lower than threshold.
                for client_id, trust in client_trusts.items():
                    if trust < self.trust_threshold and client_id not in self.removed_client_ids:
                        logging.info(f"Removing client with TRUST less than Threshold: {client_id}")
                        # add this client to the removed_clients list
                        self.removed_client_ids.add(client_id)

            logging.info(f"removed clients are : {self.removed_client_ids}")

        # select clients based on updated TRUSTS and available clients
        sorted_client_ids = sorted(client_trusts, key=client_trusts.get, reverse=True)
        selected_client_ids = sorted_client_ids

        # create training configurations for selected clients
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
            accuracy_matrix['cid'] = cid

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
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_metadata.cid),
                current_round=self.current_round,
                loss=evaluate_res.loss
            )

            if client_metadata.cid not in self.removed_client_ids:
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
