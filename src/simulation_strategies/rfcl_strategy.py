"""
RFCL: Robust Federated Learning Method Against Data and Model Poisoning Attacks 
with Heterogeneous Data Distribution

Reference
---------
Alharbi, E., Marcolino, L. S., Gouglidis, A., & Ni, Q.
"Robust Federated Learning Method Against Data and Model Poisoning Attacks 
with Heterogeneous Data Distribution."
26th European Conference on Artificial Intelligence (ECAI 2023).
https://doi.org/10.3233/FAIA230257

Algorithm Overview
------------------
RFCL is a multi-stage, Byzantine-robust federated learning strategy designed to 
mitigate both data and model poisoning attacks in highly non-IID environments. 
It utilizes a personalized model-sharing approach to quarantine malicious clients 
while enabling honest clients to converge. Each round, the server performs the 
following steps:

  1. Dimensionality Reduction & Clustering (HDBSCAN):
     Extracts and flattens the model parameters from all clients into a single matrix.
     Applies Principal Component Analysis (PCA), retaining 95% of the variance, to 
     reduce dimensionality. The reduced weights are clustered using HDBSCAN. Clients 
     flagged as explicit outliers (label -1) are quarantined and assigned their 
     cluster's model (M_cc) from the previous round.

  2. Modified Adaptive Federated Averaging (ModiAFA):
     Within each valid cluster, computes the cosine similarity between each client's 
     update and the cluster's previous center (M_cc). It calculates the mean, median, 
     and standard deviation of these similarities. Based on the skewness of the 
     distribution, clients falling outside a dynamic threshold (controlled by sensitivity ξ) 
     are marked as malicious and their weights are set to 0. The remaining valid updates 
     are aggregated to form the new cluster center (M_cc).

  3. Similarity Analysis:
     Computes a cosine similarity matrix across all newly generated cluster centers (M_cc). 
     It identifies the top `num_similar_cluster_centers` (K) most similar clusters to 
     form the trusted pool (M_best). The aggregation probabilities for these clusters 
     are normalized based on both their similarity scores and proportional client sizes. 

  4. External Aggregation & Personalization:
     Aggregates the trusted clusters (M_best) into a single, highly robust Concentrated 
     Global Model (M_c). The strategy then distributes personalized models for the next round:
       - Selected clusters (in M_best) receive the new global model M_c.
       - Un-selected clusters retain their unique, newly computed M_cc to prevent 
         cross-contamination.
       - Quarantined outliers (from HDBSCAN or ModiAFA) fall back to the M_cc of 
         their cluster from the previous round.

Key hyperparameters
-------------------------------------
- outlier_detection_sensitivity (ξ):
    The sensitivity parameter used in the ModiAFA module. It scales the standard 
    deviation when computing the cutoff threshold for intra-cluster similarities. 
    A lower value makes the algorithm more aggressive at dropping clients, while a 
    higher value is more permissive.

- num_similar_cluster_centers (K):
    The maximum number of clusters selected during Similarity Analysis to form the 
    trusted pool (M_best). This ensures that only the most dense, highly correlated 
    groups of honest clients dictate the global model trajectory (M_c).

- min_cluster_size:
    The core parameter for the HDBSCAN clustering module. It dictates the minimum 
    number of clients required to form a valid cluster. Branches smaller than this 
    are immediately discarded as noise/outliers prior to ModiAFA.
"""

import logging
import math
import os
import time
import torch
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics
    

class RFCLStrategy(fl.server.strategy.FedAvg):
    """
    RFCL aggregation strategy (ECAI 2023).

    Parameters
    ----------
    outlier_detection_sensitivity : float
        Sensitivity parameter (ξ) used in the Modified Adaptive Federated 
        Averaging (ModiAFA) module. It acts as a multiplier for the standard 
        deviation when setting the dynamic threshold for intra-cluster cosine 
        similarities. Lower values make the outlier detection more aggressive.
    num_similar_cluster_centers : int
        The maximum number of clusters (K) selected during the Similarity 
        Analysis phase. Determines how many of the most similar cluster centers 
        are retained to form the trusted pool (M_best) for final global 
        aggregation.
    min_cluster_size : int
        Core parameter used in the HDBSCAN clustering module. Dictates the 
        minimum number of clients required to form a valid cluster. Clients 
        failing to group into a cluster of at least this size are immediately 
        flagged as noise/outliers.
    remove_clients : bool
        Whether to actively quarantine flagged clients and apply personalized 
        model sharing. Set to False to bypass clustering/ModiAFA and simply 
        aggregate all clients (useful for debugging or baseline comparisons).
    begin_removing_from_round : int
        The round number at which the RFCL filtering and clustering logic 
        activates. Rounds strictly prior to this operate as a warmup period 
        using standard FedAvg, allowing clients to establish an initial 
        trajectory before dimension reduction and clustering occur.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    output_dirname : str
        Directory for log output.
    """
    
    def __init__(
        self,
        outlier_detection_sensitivity: float,
        num_similar_cluster_centers: int,
        min_cluster_size: int,
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

        self.outlier_detection_sensitivity = outlier_detection_sensitivity
        self.num_similar_cluster_centers = num_similar_cluster_centers
        self.min_cluster_size = min_cluster_size

        self.client_to_cluster: Dict[str, Any] = {}
        self.previous_mcc: Dict[Any, np.ndarray] = {}
        self.isolated_models_this_round: Dict[str, np.ndarray] = {}
        self.removed_client_ids: Set[str] = set()

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
    

    def _modi_afa(self, indices: List[int], models: List[np.ndarray], client_ids: List[str], cluster_id: int) -> np.ndarray | None:
        """
        Modified Adaptive Federated Averaging (ModiAFA)
        """
        # 1. Compute Cosine Similarities (s_i) between prev_mcc and each client model
        similarities = []
        client_prev_models = [] # Track which model each client started with
        for i in indices:
            m_i = models[i]
            real_cid = client_ids[i]
            prev_cluser_id = self.client_to_cluster[real_cid]
            prev_mcc = self.previous_mcc[prev_cluser_id]
            client_prev_models.append(prev_mcc)
            # cosine_similarity expects 2D arrays: reshape(1, -1)
            sim = cosine_similarity(prev_mcc.reshape(1, -1), m_i.reshape(1, -1))[0][0]
            similarities.append(sim)

        # Bypass samll clusters
        if len(similarities) <= 2:
            self.logger.info(f"Cluster {cluster_id} too small for ModiAFA stats. Accepting all {len(similarities)} clients.")
            # Average the models directly
            new_mcc = np.mean([models[i] for i in indices], axis=0)
            return new_mcc

        # 2. Calculate statistics for outlier detection
        mean_s = np.mean(similarities)
        median_s = np.median(similarities)
        std_s = np.std(similarities)

        # 3. Outlier Detection logic (Algorithm 2/ModiAFA description)
        weights = []
        for i, s_i in enumerate(similarities):
            is_outlier = False
            if mean_s < median_s:
                if s_i < (median_s - self.outlier_detection_sensitivity * std_s):
                    is_outlier = True
            else: # mean_s >= median_s
                if s_i > (median_s + self.outlier_detection_sensitivity * std_s):
                    is_outlier = True
            
            # If it's an outlier, set weight to 0; otherwise 1 
            weights.append(0.0 if is_outlier else 1.0)
            if is_outlier:
                idx = indices[i]
                self.logger.info(f"Outlier detected: Client {client_ids[idx]}")
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(client_ids[idx]),
                    removal_criterion=0.666,
                )
                self.removed_client_ids.add(client_ids[idx])
        self.logger.info(f"Num Clients From Cluster {cluster_id} Excluded as Outliers by ModiAFA: {weights.count(0.0)} | {len(weights)}\n")
            
        # 4. Aggregate: Weighted average of the models (excluding outliers)
        total_weight = sum(weights)
        if total_weight == 0:
            # If all are outliers, Disqualify completely failed cluster
            self.logger.info(f"Cluster {cluster_id} completely failed ModiAFA. Disqualifying cluster.")
            return None 
            
        new_mcc = np.zeros_like(client_prev_models[0])
        for i, w in enumerate(weights):
            new_mcc += (w / total_weight) * models[indices[i]]
            
        return new_mcc
    

    def _isolate_previous_mcc(self, client_id: str):
        """
        Helper function for assocating a client with its previous m_cc model if
        it is an outlier in the current round
        """
        # Look up what cluster this client started the round with
        old_cluster_id = self.client_to_cluster.get(client_id, "fallback")
        # Retrieve that actual model array from the old dictionary
        old_mcc = self.previous_mcc.get(old_cluster_id)
        if old_mcc is not None:
            # Give them a unique isolation key
            isolation_key = f"isolated_{client_id}"
            self.client_to_cluster[client_id] = isolation_key
            # Save their old model into our temporary dictionary for outliers
            self.isolated_models_this_round[isolation_key] = old_mcc
        else:
            # Ultimate safety net
            self.client_to_cluster[client_id] = "fallback"
    

    def _clustering(self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]) -> Tuple[List[np.ndarray], List]:
        """
        Clustering (Algorithm 2)
        """
        # Step 1: Extract weights from models (line 1)
        client_ids = []
        flattened_weights = []
        for client_proxy, fit_res in results:
            client_ids.append(client_proxy.cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            # Flatten the layer weights into a single 1D vector
            flat_w = self._flatten(weights)
            flattened_weights.append(flat_w)
        
        # Create matrix W (shape: [num_clients, total_parameters])
        # The weights matrix is the input to the Dimensionality Reduction (PCA) step
        W = np.stack(flattened_weights)

        # Step 2: Apply PCA to weights for dimensionality reduction (line 2)
        # Initialize and fit_transform the PCA, retain 95% of the variance
        pca = PCA(n_components=0.95)
        W_reduced = pca.fit_transform(W)
        self.logger.info(f"Applying PCA to reduce dimensionality from {W.shape[1]} to {W_reduced.shape[1]}")

        # Step 3: Apply HDBSCAN clustering (line 3-4)
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        # Fit and predict labels
        cluster_labels = clusterer.fit_predict(W_reduced)

        # Step 4: count clusters (lines 5)
        # count only "valid" (non-outlier) clusters:
        # Note: Max label + 1 gives the count of non-outlier clusters
        cluster_count = int(cluster_labels.max() + 1)

        self.isolated_models_this_round = {}    # reset dictionary for isolated models
        # Step 5: Assign models to their respective clusters (lines 6-13)
        indices = [[] for _ in range(cluster_count)]
        cluster_centres_len = np.zeros(cluster_count)
        for i, label in enumerate(cluster_labels):
            if label != -1:
                cluster_centres_len[label] += 1 # Increment size of clusers
                indices[label].append(i)        # Append model index to respective cluster
            else:
                cid = client_ids[i]
                self.logger.info(f"Outlier detected: Client {cid}")
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    removal_criterion=0.333,
                )
                self.removed_client_ids.add(cid)
                # Isolate the client's previous m_cc model so it can be used next round
                self._isolate_previous_mcc(cid)
        self.logger.info(f"Total Number of Clusters: {cluster_count}")
        self.logger.info(f"Num Clients Excluded as Outliers by HBDSCAN: {(cluster_labels == -1).sum()}")
        self.logger.info(f"After HDBSCAN Clustering, Total Client's Excluded: {len(self.removed_client_ids)} | {len(client_ids)}\n")

        # Normalization
        # cluster_centres_len represents the proportion of clients in each cluster relative to valid clustered clients
        # Calculate the number of valid clients (total clients minus outliers)
        num_outliers = (cluster_labels == -1).sum()
        valid_client_count = len(client_ids) - num_outliers
        # Normalization: ensure the weights of the valid clusters sum to 1.0
        if valid_client_count > 0:
            cluster_centres_len /= valid_client_count
        else:
            # Absolute worst-case safety net if ALL clients are outliers
            cluster_centres_len = np.zeros_like(cluster_centres_len)

        # Step 6: Compute the cluster centres by the ModiAFA method (lines 14-16)
        valid_M_cc = []
        valid_cluster_centres_len = []
        for i, ins in enumerate(indices):
            mcc = self._modi_afa(ins, flattened_weights, client_ids, i)
            if mcc is not None:
                # The new index in our valid list
                new_cluser_index = len(valid_M_cc)
                valid_M_cc.append(mcc)
                # Keep the corresponding cluster size for the probabilities later
                valid_cluster_centres_len.append(cluster_centres_len[i])
                for idx in ins:
                    real_cid = client_ids[idx]
                    # Check if ModiAFA rejected this specific clien
                    if real_cid in self.removed_client_ids:
                        self._isolate_previous_mcc(real_cid)
                    else:
                        # Map the client to the New list index
                        self.client_to_cluster[real_cid] = new_cluser_index
            else:
                # ModiAFA completely failed. Give all clients their previous mcc
                for idx in ins:
                    real_cid = client_ids[idx]
                    self._isolate_previous_mcc(real_cid)
        self.logger.info(f"After ModiAFA, Total Client's Excluded: {len(self.removed_client_ids)} | {len(client_ids)}\n")
            
        # Return the cluster centers (M_cc) and proportional sizes (cluster_centres_len) for Algorithm 3
        return valid_M_cc, valid_cluster_centres_len
    
    
    def _similarity_analysis(self, M_cc: List[np.ndarray], cluster_centres_len: List) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """
        Similarity Analysis (Algorithm 3)
        """
        # Step 1: Check for outliers (lines 1-5)
        # the number of valid, non-outlier clusters is the length of M_cc
        unoutliers = len(M_cc)

        # Step 2: Determine the number of clusters to select (line 6)
        num_select = min(self.num_similar_cluster_centers, unoutliers)
        self.logger.info(f"Selecting {num_select} Best Clusters Out of {unoutliers}")

        if num_select == 0:
            # Fallback: return empty/default values so the main server logic knows to skip aggregation this round
            return [], np.array([]), []

        # Step 3: Calculate cosine similarity (lines 7-13)
        # Generate cluster centres weights matrix X with shape (cluster_count, num_features)
        X = np.stack(M_cc)
        # Compute the cosine similarity for every pair of weights
        # This automatically creates an cluster_count x cluster_count matrix of similarities
        similarities = cosine_similarity(X)

        # Step 4: Select the most simular clusters (lines 14-21)
        best_val = -float('inf')    # Initialize to negative infinity
        indices_best = []
        best_s = []                 # Save the winning similarity scores
        # Iterate through each row in similarity matrix
        # 's' is an array of similarity scores for cluster 'i' against all others
        for s in similarities:
            # np.argsort sorts ascending, so [-num_select:] gets the indices of the largest values
            indices = np.argsort(s)[-num_select:].tolist()
            # Calculate the sum of these selected top similarities
            val = np.sum(s[indices])
            if val > best_val:
                best_val = val
                indices_best = indices
                best_s = s              # Save the scores for the normalization step

        # Step 5: Normalize the similarity scores to obtain the probabilties of selecting each model (line 22)
        # Extract the similarity scores only for the selected best clusters
        selected_sims = np.array([best_s[i] for i in indices_best])
        # Normalize the similarity scores to obtain the initial probabilities p_s
        # Equation: p_s_i = s_i / sum(s_j)
        sum_sims = np.sum(selected_sims)
        p_s = selected_sims / (sum_sims if sum_sims > 0 else 1e-9)

        # Step 6: Reconfigure Probabilities (lines 23-25)
        # Retrieve the actual models for the selected clusters
        M_best = [M_cc[i] for i in indices_best]

        # Adjust the probabilities based on the proportional size of each cluster
        selected_cluster_len = np.array([cluster_centres_len[i] for i in indices_best])
        # ps.mul(cluster_centre_len)
        p_s = p_s * selected_cluster_len
        # Normalize them again so they sum to 1.0
        sum_p_s = np.sum(p_s)
        p_s = p_s / (sum_p_s if sum_p_s > 0 else 1e-9)

        # Log Similarity-Based Exlusions
        # Find which cluster indices were NOT selected
        all_cluster_indices = set(range(len(M_cc)))
        excluded_clusters = all_cluster_indices - set(indices_best)
        num_excluded_clients = 0
        # Iterate through the mapping to find the clients in those excluded clusters
        for cid, cluster_id in self.client_to_cluster.items():
            if cluster_id in excluded_clusters:
                # Check if they weren't already excluded
                if cid not in self.removed_client_ids:
                    self.logger.info(f"Excluded: Client {cid}")
                    self.strategy_history.insert_single_client_history_entry(
                        current_round=self.current_round,
                        client_id=int(cid),
                        removal_criterion=1.0,
                    )
                    num_excluded_clients += 1
                    self.removed_client_ids.add(cid)
            else:
                if cid not in self.removed_client_ids:
                    self.strategy_history.insert_single_client_history_entry(
                        current_round=self.current_round,
                        client_id=int(cid),
                        removal_criterion=0.0,
                    )
        self.logger.info(f"Num Clients Excluded by Similarity Analysis: {num_excluded_clients}")
        self.logger.info(f"After Similarity Analysis, Total Client's Excluded: {len(self.removed_client_ids)} | {len(self.client_to_cluster)}\n")
                    
        # Return the final selected models (M_best), their probabilities (p_s), and indices
        return M_best, p_s, indices_best
    

    def _update_cluster_centres(self, M_c: np.ndarray, M_cc: List[np.ndarray], indices_best: List[int]) -> List[np.ndarray]:
        """
        Personalization (Cluster-based Model Sharing) (Algorithm 4)
        """
        # Directly iterate over the winning indices and overwrite them
        for i in indices_best:
            M_cc[i] = M_c
        return M_cc


    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        # fetch the available clients as a dictionary
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        fit_instructions = []

        # Algorithm 1: RFCL Process (lines 2-6)
        # Capture w before clients train on it — this is the initial weights
        # used in ModiAFA for first round
        # Handles round 0 or warmup rounds or no removal
        if self.current_round == 0 or not self.remove_clients or self.current_round < self.begin_removing_from_round:
            current_global_weights = fl.common.parameters_to_ndarrays(parameters)
            current_flattened_global_weights = self._flatten(current_global_weights)
            self.previous_mcc = {0: current_flattened_global_weights, "fallback": current_flattened_global_weights}
            self.client_to_cluster = {client_id: 0 for client_id in available_clients.keys()}
            # In round 0, warmup rounds, or no removal, all clients receive the initial global weights
            fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
            for client_proxy in available_clients.values():
                fit_instructions.append((client_proxy, fit_ins))
        else:
            for client_id, client_proxy in available_clients.items():
                previous_cluster_id = self.client_to_cluster.get(client_id, "fallback")
                previous_mcc = self.previous_mcc[previous_cluster_id]
                template_weights = parameters_to_ndarrays(parameters)
                unflattened_M_cc = self._unflatten(previous_mcc, template_weights)
                client_parameters = fl.common.ndarrays_to_parameters(unflattened_M_cc)
                fit_ins = fl.common.FitIns(client_parameters, {"server_round": server_round})
                fit_instructions.append((client_proxy, fit_ins))

        # Reset per-round state
        self.removed_client_ids = set()

        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        return fit_instructions


    def _standard_logging_handler(self, server_round: int, num_clients: int, removal_threshold: float, score_calculation_time_nanos: int) -> None:
        """
        Helper function for handling the standard storage and logging that needs 
        to be done before returning from aggregate_fit
        """
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=score_calculation_time_nanos,
            removal_threshold=removal_threshold,
        )

        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        if num_clients != 0:
            # ── Log filtering results ────────────────────────────────────────
            self.logger.info(
                f"Round {server_round}: filtered out Clients {self.removed_client_ids}"
            )
            self.logger.info(
                f"Round {server_round}: accepted {num_clients-len(self.removed_client_ids)}/{num_clients} "
                f"clients | filtered {len(self.removed_client_ids)}"
            )

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)
        
        # Distance clustering for distance metrics tracking
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
        for i, (client_proxy, fit_res) in enumerate(results):
            client_id = client_proxy.cid
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                absolute_distance=float(distances[i][0])
            )

        time_start = time.time_ns()
        num_clients = len(results)

        if num_clients < self.min_cluster_size or not self.remove_clients or server_round < self.begin_removing_from_round:
            # Parameter check
            if num_clients < self.min_cluster_size:
                self.logger.info(
                    f"Round {server_round}: too few clients ({num_clients}) for "
                    f"BTBCN — aggregating all"
                )
            else:
                self.logger.info(f"Round {server_round}: Warmup round - aggregating all")

            time_end = time.time_ns()

            # ── Record metrics ───────────────────────────────────────────────
            for client_proxy, _ in results:
                cid = client_proxy.cid
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    removal_criterion=0.0
                )
            self._standard_logging_handler(server_round, 0, 0.2, time_end - time_start)
            
            return super().aggregate_fit(server_round, results, failures)

        # Algorithm 1: RFCL Process (lines 8-12)
        # Step 1: Perform clusteing on client models M_i --> M_cc
        M_cc, cluster_centres_len = self._clustering(results)

        # Step 2: Select the most similar models in M_cc --> M_best
        M_best, p_s, indices_best = self._similarity_analysis(M_cc, cluster_centres_len)

        if not M_best:
            time_end = time.time_ns()

            # If all clients were marked as outliers, fallback
            self.logger.info(f"All clients/clusters were excluded! Rejecting Round {server_round} updates.")

            # Retrieve the last known good global model (saved from the previous round)
            M_g = self.previous_mcc.get("fallback")

            # Absolute worst-case safety net: if fallback isn't set yet (e.g., Round 1 crash)
            if M_g is None:
                # Empty removed clients set so there is no error during aggregate_evaluate
                self.removed_client_ids = set()
                # ── Record metrics and Log filtering results ───────────────────────────────────────────────
                self._standard_logging_handler(server_round, num_clients, 0.2, time_end - time_start)
                self.logger.error("No fallback model available. Falling back to FedAvg.")
                return super().aggregate_fit(server_round, results, failures)

            # 1. Hard reset previous_mcc to ONLY contain the fallback global model
            self.previous_mcc = {"fallback": M_g}
            
            # 2. Merge in the quarantined models to give the excluded clients their previous M_cc
            self.previous_mcc.update(self.isolated_models_this_round)

            # 3. Return the unchanged previous global model
            template_weights = parameters_to_ndarrays(results[0][1].parameters)
            unflattened_M_g = self._unflatten(M_g, template_weights)
            
            # ── Record metrics and Log filtering results ───────────────────────────────────────────────
            self._standard_logging_handler(server_round, num_clients, 0.2, time_end - time_start)
            # Empty removed clients set so there is no error during aggregate_evaluate
            self.removed_client_ids = set()
            return ndarrays_to_parameters(unflattened_M_g), {}

        # Step 3: External Aggregate M_best --> M_c
        # Combine the most trusted cluster centers into the final Concentrated Global Model (M_c)
        # np.average with the 'weights' parameter automatically multiplies each array
        # in M_best by the corresponding probability in p_s and sums them together
        M_c = np.average(M_best, weights=p_s, axis=0)

        # Step 4: Update the cluster centers of M_best with M_c
        M_cc = self._update_cluster_centres(M_c, M_cc, indices_best)

        # Step 5: Return M_c as the new global model M_g
        M_g = M_c
        template_weights = parameters_to_ndarrays(results[0][1].parameters)
        unflattened_M_g = self._unflatten(M_g, template_weights)

        time_end = time.time_ns()

        # Save state for next round
        # Map the valid cluser indices to their models
        self.previous_mcc = {i: model for i, model in enumerate(M_cc)}
        # Create the special fallback key for excluded clients
        self.previous_mcc["fallback"] = M_g
        # Merge n the isolated models - manually carries outlier client's old models forward
        self.previous_mcc.update(self.isolated_models_this_round)

        # ── Record metrics and Log filtering results ───────────────────────────────────────────────
        self._standard_logging_handler(server_round, num_clients, 0.2, time_end - time_start)
        return ndarrays_to_parameters(unflattened_M_g), {}

        
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
