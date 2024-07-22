from typing import List, Tuple, Optional, Union, Dict
from flwr.common import FitRes, Parameters, Scalar
from sklearn.cluster import KMeans
from flwr.server.strategy.aggregate import weighted_loss_avg
# Define strategy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
import torch
# Define strategy - BELOW IS THE STRATEGY CLASS CODE
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
import math as m

import matplotlib.pyplot as plt

class TrustPermanentRemovalStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            remove_clients,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.client_reputations = {}  # Stores the reputation of each client
        self.current_round = 0  # Initialize a round counter
        self.client_reputations_history = {}
        self.client_trust_history = {}
        self.client_trusts = {}
        self.removed_clients = set()
        self.client_accuracy_history = {}

        self.remove_clients = remove_clients
        self.total_loss_history_record = {'remove_clients': self.remove_clients, 'rounds_history': {}}

        self.client_cluster_distance_history = {}

    # HELPER FUNCTION
    def calculate_reputation(self, client_id, truth_value):
        if self.current_round == 1:
            return truth_value
        else:
            prev_reputation = self.client_reputations.get(client_id, 0)
            return self.update_reputation(prev_reputation, truth_value, self.current_round)

    # HELPER FUNCTION
    def update_reputation(self, prev_reputation, truth_value, current_round):
        # Reputation update logic
        alpha = 0.75  # Can be adjusted as needed
        if truth_value >= 0.5:
            updated_reputation = (prev_reputation + truth_value) - (prev_reputation / current_round)
        else:
            # Adjust reputation for truth values less than 0.5
            temp = -(1 - (truth_value * (prev_reputation / current_round)))
            updated_reputation = (prev_reputation + truth_value) - np.exp(temp)
        updated_reputation = alpha * updated_reputation + (1 - alpha) * prev_reputation

        if updated_reputation > 1.0:
            return 1.0
        elif updated_reputation < 0.0:
            return 0.0

        return updated_reputation[0]

    # HELPER FUNCTION
    def calculate_trust(self, client_id, reputation, d):
        '''
        Function to get previous rounds trust value and calculate trust
        '''
        if self.current_round == 1:
            prev_trust = 0
        else:
            prev_trust = self.client_trusts.get(client_id, 0)
        return self.update_trust(prev_trust, reputation, d)

    def update_trust(self, prev_trust, reputation, d):
        """
            Helper method that calculates trust
            Based on reputation value of a client
        """
        alpha = 0.85
        trust = m.sqrt(m.pow(reputation, 2) + m.pow(d, 2)) - m.sqrt(m.pow(1 - reputation, 2) + m.pow(1 - d, 2))
        trust = alpha * trust + (1 - alpha) * prev_trust

        if trust > 1.0:
            trust = 1.0
        if trust < 0.0:
            trust = 0.0
        return trust

    # Helper function to get Cosine Similarity between 2 Tensors:
    def cosine_similarity(self, tensor1, tensor2):
        dot_product = torch.dot(tensor1, tensor2)
        norm1 = torch.norm(tensor1)
        norm2 = torch.norm(tensor2)
        return dot_product / (norm1 * norm2)

    # FIT FUNCTION

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Increment the round counter
        self.current_round += 1
        aggregate_clients = []
        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_clients:
                aggregate_clients.append(result)

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, aggregate_clients, failures)

        ##CLUSTERING
        # Extract data for clustering
        clustering_param_data = []
        for client_proxy, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_tensor_list = [torch.Tensor(arr) for arr in client_params]
            flattened_param_list = [param.flatten() for param in params_tensor_list]
            param_tensor = torch.cat(flattened_param_list)
            # Extract mean of weights and bias of the last layer (fc3)
            clustering_param_data.append(param_tensor)
        # Perform clustering
        X = np.array(clustering_param_data)

        plt.scatter(X[:, 0], X[:, 1], s=50)

        kmeans = KMeans(n_clusters=1, init='k-means++').fit(X)

        y_kmeans = kmeans.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title(f'Round {self.current_round}')
        plt.show()

        distances = kmeans.transform(X) ** 2
        normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        # print(f'Aggregation round: {server_round}\nNormalized distances: {normalized_distances}')

        # Calculate reputation for each client
        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            truth_value = 1 - normalized_distances[i]
            new_reputation = self.calculate_reputation(client_id, truth_value)
            # add trust computation
            new_trust = self.calculate_trust(client_id, new_reputation, truth_value)
            self.client_reputations[client_id] = new_reputation
            # store trust for this client
            self.client_trusts[client_id] = new_trust
            print(
                f'Aggregation round: {server_round}'
                f'Client ID: {client_id}'
                f'Truth Value: {truth_value}'
                f'Reputation: {new_reputation}'
                f'Trust: {new_trust}'
                f'Normalized Distance: {normalized_distances[i]}'
            )

            if client_id not in self.client_cluster_distance_history:
                self.client_cluster_distance_history[client_id] = []

            self.client_cluster_distance_history[client_id].append(normalized_distances[i][0])

        # Update the history of reputations
        for client_id, reputation in self.client_reputations.items():
            if client_id not in self.client_reputations_history:
                self.client_reputations_history[client_id] = []
            self.client_reputations_history[client_id].append(reputation)

        # Update the history of trusts
        for client_id, trust in self.client_trusts.items():
            if client_id not in self.client_trust_history:
                self.client_trust_history[client_id] = []
            self.client_trust_history[client_id].append(trust)

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        # Trust Threshold
        trust_threshold = 0.15
        # Fetch the available clients as a dictionary
        available_clients = client_manager.all()  # Dictionary with client IDs as keys and RayActorClientProxy objects as values

        # In the Warmup rounds, select all clients
        if self.current_round <= 3:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

            # Fetch the Trust and Reputation scores for all available clients
        client_trusts = {client_id: self.client_trusts.get(client_id, 0) for client_id in available_clients.keys()}

        if self.remove_clients:
            # In the first round after warmup, remove the client with the lowest TRUST
            if self.current_round == 4:
                lowest_trust_client = min(client_trusts, key=client_trusts.get)
                print(f"Removing client with lowest TRUST: {lowest_trust_client}")
                # Add this client to the removed_clients list
                self.removed_clients.add(lowest_trust_client)
            else:
                # remove clients with trust lower than threshold.
                for client_id, trust in client_trusts.items():
                    if trust < trust_threshold and client_id not in self.removed_clients:
                        print(f"Removing client with TRUST less than Threshold: {client_id}")
                        # Add this client to the removed_clients list
                        self.removed_clients.add(client_id)

            print(f"removed clients are : {self.removed_clients}")

        # Select clients based on updated TRUSTS and available clients
        sorted_client_ids = sorted(client_trusts, key=client_trusts.get, reverse=True)
        selected_client_ids = sorted_client_ids

        # Create training configurations for selected clients
        fit_ins = fl.common.FitIns(parameters, {})
        return [(available_clients[cid], fit_ins) for cid in selected_client_ids if cid in available_clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        print(f'%%%%%%%%%% AGGREGATION ROUND {server_round} %%%%%%%%%%')
        self.client_accuracy_history[server_round] = []
        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics
            accuracy_matrix['cid'] = cid
            self.client_accuracy_history[server_round].append(accuracy_matrix)

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            if client_metadata.cid not in self.removed_clients:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)

        for result in results:
            print(f'Client ID: {result[0].cid}')
            print(f'Metrics: {result[1].metrics}')
            print(f'Loss: {result[1].loss}')

        metrics_aggregated = {}

        self.total_loss_history_record['rounds_history'][server_round] = loss_aggregated

        print(
            f'Round: {server_round} '
            f'Number of aggregated clients:{number_of_clients_in_loss_calc} '
            f'Aggregated loss: {loss_aggregated} '
        )

        return loss_aggregated, metrics_aggregated
