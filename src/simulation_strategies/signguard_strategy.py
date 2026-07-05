"""
SignGuard: Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering

Reference
---------
Xu, J., Huang, S. L., Song, L., & Lan, T.
"Byzantine-robust Federated Learning through Collaborative Malicious Gradient Filtering."
2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS), 2022.
https://doi.org/10.1109/ICDCS54860.2022.00120

Algorithm Overview
------------------
SignGuard is a robust aggregation framework that applies two parallel filtering 
mechanisms to detect and isolate malicious (Byzantine) gradients before aggregation.
It assumes that the majority of participating clients in any round are honest. 
Each round, the server performs the following steps:

  1. Gradient Setup:
     Calculates the L2-norm and the element-wise sign (-1, 0, 1) of each gradient.

  2. Norm-based Filtering (S1):
     Finds the median gradient norm (M) across all clients. Calculates the ratio 
     of each client's norm to the median (||g_t^i|| / M). Clients whose ratios 
     fall outside a defined bound [L, R] are filtered out. This protects against 
     magnitude-scaling attacks.

  3. Sign-based Clustering (S2):
     Randomly samples a subset of coordinates from the gradient vectors. For each 
     client, computes the proportion of positive, zero, and negative signs within 
     this subset, resulting in a 3D feature vector. Applies Mean-Shift clustering 
     to these features. The cluster containing the largest number of clients is 
     designated as the benign majority. This protects against sign-flipping and 
     direction-based attacks.

  4. Aggregation with Norm Clipping:
     Intersects the sets from the two filters (S_prime = S1 ∩ S2) to form the 
     final trusted set. Aggregates the trusted gradients using mean aggregation, 
     but applies a norm clipping factor: min(1, M / ||g_t^i||). This ensures 
     that any malicious client that bypasses the filters has a strictly bounded 
     impact on the global update.

Key hyperparameters (from the paper/implementation)
-------------------------------------
- norm_lower_bound (L) & norm_upper_bound (R):
    The acceptable range for the ratio of a client's gradient norm to the median 
    norm (M). The paper suggests a loose lower bound (e.g., 0.1) to retain clients 
    with small updates, and a stricter upper bound (e.g., 1.5 - 3.0) to aggressively 
    filter out abnormally large gradients typically associated with poisoning.

- coordinate_subset_ratio:
    The fraction of total model parameters randomly sampled to calculate the 
    sign proportions for clustering. Using a subset rather than all parameters 
    adds stochasticity (improving robustness against adaptive attacks) and 
    greatly reduces computational overhead.

- bandwidth / quantile (Mean-Shift):
    Determines the window size for Mean-Shift clustering. In highly non-IID 
    environments, benign gradients naturally diverge, requiring a larger bandwidth 
    (or a higher quantile estimation, e.g., 0.6 - 0.7) to prevent the algorithm 
    from fragmenting the honest majority into many small clusters and over-filtering.
"""

import logging
import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from sklearn.cluster import MeanShift, estimate_bandwidth

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics
    

class SignGuardStrategy(fl.server.strategy.FedAvg):
    """
    SignGuard aggregation strategy (ICDCS 2022).

    Parameters
    ----------
    norm_lower_bound : float
        Lower threshold for the Norm-based Filter. Defines the minimum acceptable 
        ratio of a client's gradient norm to the median norm (||g_t^i|| / M). 
        The paper recommends a loose lower bound to retain clients with naturally 
        small updates. (paper recommended: ~0.1).
    norm_upper_bound : float
        Upper threshold for the Norm-based Filter. Defines the maximum acceptable 
        ratio of a client's gradient norm to the median norm (||g_t^i|| / M). 
        A stricter upper bound is used to aggressively filter out abnormally large 
        gradients indicative of magnitude poisoning attacks. (paper recommended: 1.5 to 3.0).
    coordinate_subset_ratio : float
        Fraction of total gradient coordinates randomly sampled to calculate the 
        sign statistics (proportions of positive, negative, and zero values) for 
        the Sign-based Clustering step. Reduces computational overhead and adds 
        stochasticity against adaptive attacks. Range is (0, 1].
    bandwidth_quantile : float
        Quantile used to estimate the bandwidth for the Mean-Shift clustering algorithm. 
        Controls the window size for discovering clusters. In highly non-IID 
        environments, a higher quantile (e.g., 0.6 - 0.7) prevents the algorithm 
        from over-fragmenting the honest benign majority into many small clusters. 
        (default sklearn estimation uses 0.3).
    remove_clients : bool
        Whether to actually exclude filtered clients from the final aggregation.
        Set to False to execute the SignGuard algorithm and log filtering results 
        without actually dropping clients from the global update.
    begin_removing_from_round : int
        Warmup rounds before active filtering begins. During warmup, the server 
        still applies norm clipping to all clients for stability and calculates 
        filtering logic for logging, but no clients are explicitly removed from 
        the trusted set.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric and participation recording.
    output_dirname : str
        Directory for log output.
    """
    
    def __init__(
        self,
        norm_lower_bound: float,
        norm_upper_bound: float,
        coordinate_subset_ratio: float,
        bandwidth_quantile: float,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        output_dirname,
        *args,
        **kwargs   
    ):
        super().__init__(*args, **kwargs)
        self.current_round = 0

        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.strategy_history = strategy_history

        self.norm_lower_bound = norm_lower_bound
        self.norm_upper_bound = norm_upper_bound
        self.coordinate_subset_ratio = coordinate_subset_ratio
        self.bandwidth_quantile = bandwidth_quantile

        self.removed_client_ids: Set[str] = set()
        self.current_flatten_global_weights: Optional[np.ndarray] = None  # w (lagged)

        # Create a logger
        self.logger = logging.getLogger(f"pid_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        # Create handlers
        out_dir = output_dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False


    @staticmethod
    def _flatten(weights: List[np.ndarray]) -> np.ndarray:
        """Concatenate all weight arrays into a single 1-D float32 vector."""
        return np.concatenate([w.flatten().astype(np.float32) for w in weights])
    

    @staticmethod
    def _unflatten(flat_weights: np.ndarray, template_weights: List[np.ndarray]) -> List[np.ndarray]:
        """Reshapes a 1D array back into a list of arrays matching the template shapes."""
        unflattened_weights = []
        current_index = 0
        for template in template_weights:
            layer_size = template.size
            # Slice the 1D array and reshape it to the original layer's shape
            layer_weights = flat_weights[current_index : current_index + layer_size].reshape(template.shape)
            unflattened_weights.append(layer_weights)
            current_index += layer_size
        return unflattened_weights
    

    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        # fetch the available clients as a dictionary
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        # Capture w before clients train on it — this is the lagged reference
        # used to compute the pseudo-gradient (update) for all clients
        current_global_weights = fl.common.parameters_to_ndarrays(parameters)
        self.current_flatten_global_weights = self._flatten(current_global_weights)
            
        # Reset per-round state
        self.removed_client_ids = set()

        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(client, fit_ins) for client in available_clients.values()]

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)
        
        time_start = time.time_ns()

        # Pre-processing Step - Extract, flatten, and stack weights
        client_ids = []
        flattened_weights = []
        for client_proxy, fit_res in results:
            client_ids.append(client_proxy.cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            # Flatten the layer weights into a single 1D vector
            flat_w = self._flatten(weights)
            flattened_weights.append(flat_w)
        # Convert the list of 1D arrays into a 2D NumPy array
        # Shape: [num_clients, total_parameters]
        client_weights_matrix = np.array(flattened_weights)
        # Compute the pseudo-gradient (update) for all clients simultaneously via broadcasting
        # Gives the matrix W (shape: [num_clients, total_parameters])
        W = client_weights_matrix - self.current_flatten_global_weights
        num_clients, num_params = W.shape


        ##### Algorithm 2: SignGuard Function #####


        # Step 0: Initial Setup
        # Get L2-norm and element-wise sign of each gradient
        # 1. Compute the L2-norm of each client's gradient vector
        # Specifying axis=1, can compute the norm across the columns (parameters) for each row (client)
        gradient_norms = np.linalg.norm(W, axis=1)

        # 2. Compute the element-wise sign of each gradient
        # Converts all positive numbers to 1, negative numbers to -1, and zeros to 0
        gradient_signs = np.sign(W)
        

        # Step 1: Norm-based Filtering 
        # Get the median of gradient norms M
        M = np.median(gradient_norms)

        # Calculate the ratio |g_t^i||/M for each client i
        norm_ratios = gradient_norms / M

        # Create a boolean mask for clients that satisfy L <= ||g_t^i||/M <= R
        S1_mask = (self.norm_lower_bound <= norm_ratios) & (norm_ratios <= self.norm_upper_bound)

        norm_filtered_client_ids = []
        for i in range(num_clients):
            self.logger.info(
                f"Round {self.current_round} | Client {client_ids[i]} | "
                f"Norm Ratio: {norm_ratios[i]} | Norm Lower Bound: {self.norm_lower_bound} | Norm Upper Bound: {self.norm_upper_bound}"
            )
            if not S1_mask[i]:
                norm_filtered_client_ids.append(client_ids[i])
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(client_ids[i]),
                    removal_criterion=0.333,
                )
        
        self.logger.info(f"Round {self.current_round} | Clients Filtered by Norm: {norm_filtered_client_ids}")


        # Step 2: Sign-based Clustering
        # Randomly select a subset of gradient coordinates
        num_samples = int(self.coordinate_subset_ratio * num_params)
        sampled_indices = np.random.choice(num_params, num_samples, replace=False)
        # Extract the signs of the susbet (-1, 0, 1)
        subset_signs = gradient_signs[:, sampled_indices]

        # Compute sign statistics on selected coordinates for each gradient as features
        # Need the proportion of positive, zero, and negative signs for each client
        features = np.zeros((num_clients, 3))
        # Vectorized counting of proportions across the parameter axis (axis=1)
        features[:,0] = np.mean(subset_signs > 0, axis=1)   # Proportion of positives
        features[:,1] = np.mean(subset_signs == 0, axis=1)  # Proportion of zeros
        features[:,2] = np.mean(subset_signs < 0, axis=1)   # Proportion of negatives

        # Estimate bandwidth with a HIGHER quantile. 
        # The default is 0.3. Increasing it (e.g., to 0.5, 0.6, or 0.7) makes the window larger, 
        # forcing Mean-Shift to group more spread-out points into a single cluster.
        bw = estimate_bandwidth(features, quantile=self.bandwidth_quantile)
        # Fallback: if the variance is so small that bw is 0, give it a tiny static window
        if bw == 0:
            bw = 0.05

        # Train a Mean-Shift clustering model
        # MeanShift automatically discovers the number of clusters
        clustering = MeanShift(bandwidth=bw).fit(features)
        cluster_labels = clustering.labels_

        # Choose the cluster with the most elements as S2
        # np.bincount counts the frequencies of each cluster label
        largest_cluster_label = np.bincount(cluster_labels).argmax()
        # Create a boolean mask for clients in the largest cluster
        S2_mask = (cluster_labels == largest_cluster_label)

        cluster_filtered_client_ids = []
        for i in range(num_clients):
            if not S2_mask[i]:
                cluster_filtered_client_ids.append(client_ids[i])
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(client_ids[i]),
                    removal_criterion=0.666,
                )
        self.logger.info(f"Round {self.current_round} | Clients Filtered by Sign-Based Clustering: {cluster_filtered_client_ids}")


        # Step 3: Aggregation
        # Paper uses mean aggregation with norm clipping (what is used below)
        if not self.remove_clients or server_round < self.begin_removing_from_round:
            S_prime_mask = np.ones(num_clients, dtype=bool)
        else:
            S_prime_mask = S1_mask & S2_mask 

        if not np.any(S_prime_mask):
            self.logger.info(
                f"Round {server_round}: no clients accepted — falling back to all"
            )
            return super().aggregate_fit(server_round, results, failures)
        
        # Extract gradients for the trusted set
        trusted_W = W[S_prime_mask]
        trusted_norms = gradient_norms[S_prime_mask]

        # Apply Norm Clipping: min(1, M / ||g_t^i||)
        clipping_factors = np.minimum(1, M / trusted_norms).reshape(-1, 1)
        clipped_gradients = trusted_W * clipping_factors

        # Compute the global gradient update (mean of clipped gradients)
        global_update = np.mean(clipped_gradients, axis=0)

        # Apply the update to the global model
        new_flattened_weights = self.current_flatten_global_weights + global_update
        template_weights = parameters_to_ndarrays(results[0][1].parameters)
        unflattened_new_weights = self._unflatten(new_flattened_weights, template_weights)

        time_end = time.time_ns()

        accepted_clients = set()
        for i in range(num_clients):
            logged_norm_ratio = norm_ratios[i]
            if S_prime_mask[i]:
                accepted_clients.add(client_ids[i])
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(client_ids[i]),
                    removal_criterion=0.0,
                )
            else:
                self.removed_client_ids.add(client_ids[i])
                logged_norm_ratio = -logged_norm_ratio

            if S1_mask[i] == 0 and S2_mask[i] == 0:
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(client_ids[i]),
                    removal_criterion = 1.0,
                )
            
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_ids[i]),
                absolute_distance=logged_norm_ratio,
            )
                
        # ── Record metrics ───────────────────────────────────────────────
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
            removal_threshold=0.1
        )
        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        # ── Log filtering results ────────────────────────────────────────
        self.logger.info(
            f"Round {server_round}: All Removed Clients {self.removed_client_ids}"
        )

        self.logger.info(
            f"Round {server_round}: accepted {len(accepted_clients)}/{num_clients} "
            f"clients | filtered {len(self.removed_client_ids)}"
        )

        # ── Update global model ──────────────────────────────────────────
        return ndarrays_to_parameters(unflattened_new_weights), {}

        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info(
            '\n' + '-' * 50 + f' AGGREGATION ROUND {server_round} ' + '-' * 50
        )

        for client_result in results:
            cid = client_result[0].cid
            metrics = client_result[1].metrics
            metrics['cid'] = cid
            store_evaluation_metrics(
                clinet_id=int(cid),
                current_round=self.current_round,
                metrics=metrics, 
                strategy_history=self.strategy_history
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
                if self.strategy_history.strategy_config.model_type == "lstm":
                    aggregate_value.append((evaluate_res.metrics['num_elements'], evaluate_res.loss))
                else:
                    aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=self.removed_client_ids,
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.begin_removing_from_round,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f'Round: {server_round} | '
            f'Aggregated clients: {number_of_clients_in_loss_calc} | '
            f'Loss: {loss_aggregated:.4f}'
        )

        return loss_aggregated, metrics_aggregated
