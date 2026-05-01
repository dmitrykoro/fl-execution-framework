"""
FedDMC: Efficient and Robust Federated Learning via Detecting Malicious Clients

Reference
---------
Mu, X., Cheng, K., Shen, Y., Li, X., Chang, Z., Zhang, T., & Ma, X.
"FedDMC: Efficient and Robust Federated Learning via Detecting Malicious Clients."
IEEE Transactions on Dependable and Secure Computing (TDSC), Vol. 21, No. 6, 2024.
https://doi.org/10.1109/TDSC.2024.3372634

Algorithm Overview
------------------
FedDMC is a *pre-filtering* aggregation strategy that wraps a standard aggregator
(e.g., FedAvg). It assumes adversaries control less than half of the participating
clients in any given round. Each round, the server performs the following steps:

  1. Dimensionality Reduction (DR):
     Extracts and flattens the model parameters from all clients into a single matrix.
     Applies Principal Component Analysis (PCA) to project these high-dimensional
     updates into a very low-dimensional space (e.g., k=5 components). This retains
     primary features while massively reducing the computational cost of clustering.

  2. Binary Tree-Based Clustering with Noise (BTBCN):
     Builds a hierarchical agglomerative clustering tree (dendrogram) using the
     PCA-reduced vectors. It recursively inspects the branches: if a branch contains
     fewer clients than ``min_cluster_size``, it is pruned and marked as noise/malicious.
     The remaining valid tree is split into two clusters. The larger cluster is labeled
     benign (1), and the smaller cluster (and noise) is labeled malicious (0).

  3. Self-Ensemble Detection Correction (SEDC):
     Maintains an exponentially weighted moving average (trust score) for each client 
     across rounds to prevent false positives from random noise.
         S_score_i^t = α * S_score_i^{t-1} + (1 - α) * current_detection_i^t
     During a warmup period (round ≤ ``begin_removing_from_round``), trust scores 
     are updated but no filtering occurs.

  4. Robust Aggregation:
     Flags clients whose updated trust score falls strictly below the ``threshold``
     (typically 0.5). Flagged clients are excluded, and the remaining benign updates 
     are aggregated using standard FedAvg.

Key hyperparameters (from the paper)
-------------------------------------
- num_principal_components (k):
    The number of dimensions retained after PCA. The paper rigorously evaluated 
    k ∈ {2, 5, 10, 20, 100} and found that k = 5 provides the optimal balance of 
    feature retention and spatial separation for clustering. Constraint: k ≤ min(N, d).

- min_cluster_size:
    The noise threshold used in the BTBCN module. If a tree branch has fewer clients 
    than this value, it is discarded as an outlier. The paper recommends 3 as the 
    default to avoid accidentally filtering out small groups of honest clients with 
    highly non-IID data.

- ensemble_alpha (α):
    Controls the weight of the historical trust score vs. the current round's 
    detection result. Range is [0, 1). The paper states the algorithm is relatively 
    insensitive to this parameter and uses 0.5 as a balanced default.

- threshold:
    The cutoff point for the SEDC trust score. Because BTBCN outputs 1 (benign) 
    or 0 (malicious) and scores initialize at 0.5, a threshold of 0.5 is the 
    natural mathematical pivot point to drop malicious clients.
"""

import logging
import math
import os
import time
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics
    

class FedDMCStrategy(fl.server.strategy.FedAvg):
    """
    FedDMC aggregation strategy (TDSC 2024).

    Parameters
    ----------
    ensemble_alpha : float
        Controls the weight given to a client's historical trust score versus 
        the current round's detection result in the Self-Ensemble Detection 
        Correction (SEDC) module. Range is [0, 1). (paper default: 0.5).
    num_principal_components : int
        Number of dimensions retained after applying PCA to the flattened 
        model updates. Reduces computational overhead for the clustering step. 
        (paper default: 5). Must be ≤ min(clients_per_round, total_parameters).
    min_cluster_size : int
        Noise threshold used in the BTBCN module. Hierarchical tree branches 
        containing fewer clients than this value are pruned as outliers/noise. 
        (paper default: 3).
    threshold : float
        Removal threshold for the SEDC trust score. Clients whose trust score 
        falls strictly below this value are classified as malicious and 
        excluded from aggregation. (paper default: 0.5).
    remove_clients : bool
        Whether to actually exclude flagged clients from aggregation.
        Set to False to calculate and log trust scores without modifying 
        the aggregation (useful for debugging / analysis).
    begin_removing_from_round : int
        Warmup rounds before filtering is activated. During warmup, trust 
        scores are still calculated and accumulated via SEDC, but no clients 
        are actually removed from the FedAvg step.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    output_dirname : str
        Directory for log output.
    """
    
    def __init__(
        self,
        ensemble_alpha: float,
        num_principal_components: int,
        min_cluster_size: int,
        threshold: float,
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

        self.ensemble_alpha = ensemble_alpha
        self.num_principal_components = num_principal_components
        self.min_cluster_size = min_cluster_size
        self.threshold = threshold

        self.trust_scores: dict[str, float] = {}
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
    
    
    def _get_leaves(self, node: int, Z: np.ndarray, n: int) -> list:
        """Helper to recursively find all original client indices under a given node."""
        if node < n:
            return [node]
        left_child = int(Z[node - n, 0])
        right_child = int(Z[node - n, 1])
        return self._get_leaves(left_child, Z, n) + self._get_leaves(right_child, Z, n)


    def _btbcn_clustering(self, W_reduced: np.ndarray, client_ids: List[str]) -> np.ndarray:
        """
        Binary Tree-Based Clustering With Noise (BTBCN) (Algorithm 2)

        Args:
            W_reduced: The PCA-reduced matrix of client updates.
            client_ids: The list of client IDs corresponding to the rows of W_reduced.

        Returns:
            S: A binary numpy array where 1 represents benign and 0 represents malicious.
        """

        n = len(W_reduced)
        S = np.ones(n, dtype=int) # Output

        # --------------------------------
        # Buld Binary Tree (Lines 1-6)
        # scipy's linkage builds the tree using Euclidean distance
        # and merges the closest clusters iteratively.
        # Z is an (n-1) x 4 matrix representing the binary tree.
        # 4 columns n any row i of matrix Z represent:
        #   Z[i, 0] = index of left child
        #   Z[i, 1] = index of right child
        #   Z[i, 2] = distance between left and right child
        #   Z[i, 3] = cluster size
        # ---------------------------------
        Z = linkage(W_reduced, method='ward', metric='euclidean')

        # --------------------------------
        # Condensed Binary Tree (Lines 7-18)
        # ---------------------------------
        r = 2*n-2 # Line 7: Root node index in scipy linkage
        outliers = []
        while True:
            # Line 8/18: NodeSplitting(r)
            if r < n:
                break # We reached a single leaf, can't split further

            Lchild = int(Z[r-n, 0]) # Left child index in scipy linkage
            Rchild = int(Z[r-n, 1]) # Right child index in scipy linkage

            # Determine sizes of the children
            L_size = 1 if Lchild < n else Z[Lchild-n, 3]
            R_size = 1 if Rchild < n else Z[Rchild-n, 3]

            # Line 9: Condition for 'is outliers' based on min_cluster_size
            L_is_outlier = L_size < self.min_cluster_size
            R_is_outlier = R_size < self.min_cluster_size

            if L_is_outlier and R_is_outlier:
                # Line 10-12: Mark Lchild as outlier, move root to Rchild
                outliers.extend(self._get_leaves(Lchild, Z, n))
                r = Rchild
            elif R_is_outlier and not L_is_outlier:
                # Line 13-15: Mark Rchild as outlier, move root to Lchild
                outliers.extend(self._get_leaves(Rchild, Z, n))
                r = Lchild
            elif L_is_outlier and R_is_outlier:
                # Both are outliers (not handled in algorithm, but should handle as edge case)
                outliers.extend(self._get_leaves(Lchild, Z, n))
                outliers.extend(self._get_leaves(Rchild, Z, n))
                break
            else:
                # Line 9 condtion fails (neither is an outlier), stop traversing
                break
            
            # Line 16-17: if outliers >= ⌊ N/2 ⌋ then break
            if len(outliers) >= n//2:
                self.logger.warning(f"Outliers exceeded N/2 ({len(outliers)} >= {n//2}). Breaking.")
                break

        # --------------------------------
        # Line 19: ClientsEvaluation(Lchild, Rchild, outlier)
        # Evaluate the remaining components to assign S
        # ---------------------------------
        # Any identified outliers are marked as malicious (0)
        for idx in outliers:
            S[idx] = 0

        # The remaining valid tree splits into Lchild and Rchild
        if r >= n:  # Ensure we are at a brnach, not a single leaf
            L_leaves = self._get_leaves(Lchild, Z, n)
            R_leaves = self._get_leaves(Rchild, Z, n)
            
            # Generally, the larger cluster is assumed to be benign
            if len(L_leaves) > len(R_leaves):
                benign_leaves = L_leaves
                malicious_leaves = R_leaves
            else:
                benign_leaves = R_leaves
                malicious_leaves = L_leaves

            # Assign benign (1) and malicious (0) labels
            for idx in benign_leaves:
                S[idx] = 1
            for idx in malicious_leaves:
                S[idx] = 0
        else:
            # Fallback if the tree condensed down to a single point
            if r not in outliers:
                S[r] = 1

        return S


    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        # fetch the available clients as a dictionary
        available_clients = client_manager.all()  # dictionary with client IDs as keys and RayActorClientProxy objects as values

        if self.current_round == 0:
            # Self-Ensemble Detection Correction (SEDC) trust scores are initialized to 0.5
            #   at round 0
            for client_id in available_clients.keys():
                self.trust_scores[client_id] = 0.5
            
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

        # Pre-processing Step - Extract, flatten, and stack weights
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

        num_clients = len(client_ids)
        # Parameter check
        if num_clients < self.min_cluster_size:
            self.logger.info(
                f"Round {server_round}: too few clients ({num_clients}) for "
                f"BTBCN — aggregating all"
            )
            time_end = time.time_ns()
            self.strategy_history.insert_round_history_entry(
                score_calculation_time_nanos=time_end - time_start,
                removal_threshold=0.0,
            )
            for cid in client_ids:
                self.strategy_history.insert_single_client_history_entry(
                    current_round=self.current_round,
                    client_id=int(cid),
                    removal_criterion=0.0,
                )
            return super().aggregate_fit(server_round, results, failures)
        

        # Algorithm 1, lines 7-11
        # Step 1: Dimensionality Redecuton Module - Principal Component Analysis (PCA)
        # Safety check: PCA cannot have more components than the minmum of samples or features.
        # So, 'k' must be less than or equal to the minimum of samples and features.
        n_samples, n_features = W.shape
        actual_k = min(self.num_principal_components, n_samples, n_features)
        self.logger.info(f"Applying PCA to reduce dimensionality from {n_features} to {actual_k}")
        
        # Initialize and fit_transform the PCA
        pca = PCA(n_components=actual_k)
        W_reduced = pca.fit_transform(W)


        # Step 2: Binary Tree-Based Clustering With Noise (BTBCN) for classifying malicious and benign clients
        S_t = self._btbcn_clustering(W_reduced, client_ids)
        malicious_count_this_round = np.sum(S_t == 0)
        benign_count_this_round = np.sum(S_t == 1)
        self.logger.info(f"BTBCN identified {malicious_count_this_round} malicious clients | {benign_count_this_round} benign clients")


        # Step 3: Self-Ensemble Detection Correction (SEDC) for history based detection and removal
        benign_results = []
        for i, (client_proxy, fit_res) in enumerate(results):
            client_id = client_proxy.cid
            current_detection = S_t[i]

            # Retrieve historical trust score
            historical_score = self.trust_scores.get(client_id, 0.5)

            # SEDC Formula: update the trust score
            updated_score = self.ensemble_alpha * historical_score + (1.0 - self.ensemble_alpha) * current_detection
            self.trust_scores[client_id] = updated_score

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=float(updated_score),
            )
            self.logger.info(
                f"Round {self.current_round} | Client {client_id} | "
                f"Current Detection: {current_detection} | Historical Score: {historical_score} | "
                f"Updated Score: {updated_score} | Removal Threshold: {self.threshold}"
            )

            # Classification based on threshold
            if updated_score >= self.threshold or not self.remove_clients or server_round < self.begin_removing_from_round:
                benign_results.append((client_proxy, fit_res))
            else:
                self.removed_client_ids.add(client_id)

        time_end = time.time_ns()

        # ── Record metrics ───────────────────────────────────────────────
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
            removal_threshold=self.threshold,
        )
        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        # ── Log filtering results ────────────────────────────────────────
        self.logger.info(f"REMOVAL THRESHOLD: {self.threshold}")
        for cid in self.removed_client_ids:
            self.logger.info(
                f"Round {server_round}: filtered out Client {cid}"
            )

        self.logger.info(
            f"Round {server_round}: accepted {len(benign_results)}/{num_clients} "
            f"clients | filtered {len(self.removed_client_ids)}"
        )

        # FINAL STEP: Robust Aggregation (FedAvg on benign clients)
        if not benign_results:
            self.logger.info(
                f"Round {server_round}: no clients accepted — falling back to all"
            )
            return super().aggregate_fit(server_round, results, failures)

        # Use FedAvg over the accepted clients
        return super().aggregate_fit(server_round, benign_results, failures)

        
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
