"""
FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning
Attacks in Federated Learning.

Reference
---------
Fereidooni, H., Pegoraro, A., Rieger, P., Dmitrienko, A., & Sadeghi, A.-R.
"FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning
Attacks in Federated Learning."  NDSS 2024.
https://www.ndss-symposium.org/ndss-paper/freqfed-a-frequency-analysis-based-approach-for-mitigating-poisoning-attacks-in-federated-learning/

Implementation of Algorithm 1 from the paper:
  1. Flatten client weights → reshape to square matrix → 2-D DCT.
  2. Extract low-frequency triangle: keep V[i,j] where i+j ≤ ⌊N/2⌋.
  3. Compute pairwise cosine-distance matrix over low-freq vectors.
  4. Cluster with HDBSCAN (metric='precomputed').
  5. Select the largest cluster → simple-average only those clients.
"""

import logging
import math
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from scipy.fft import dctn
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics


class FreqFedStrategy(fl.server.strategy.FedAvg):
    """
    FreqFed aggregation strategy.

    Parameters
    ----------
    min_cluster_size : int
        Minimum cluster size for HDBSCAN (default 2; not specified in paper, sklearn default is 5).
    remove_clients : bool
        Whether to exclude filtered clients from aggregation.
    begin_removing_from_round : int
        Warmup rounds before filtering is active.
    strategy_history : SimulationStrategyHistory
        Shared history object for metric recording.
    output_dirname : str
        Directory for log output.
    """

    def __init__(
        self,
        min_cluster_size: int,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        output_dirname: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.min_cluster_size = max(2, min_cluster_size)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.strategy_history = strategy_history

        # Round tracking
        self.current_round: int = 0
        self.removed_client_ids: Set[str] = set()

        # Logger (same pattern as other strategies)
        self.logger = logging.getLogger(f"freq_fed_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)
        os.makedirs(output_dirname, exist_ok=True)
        self.logger.addHandler(logging.FileHandler(f"{output_dirname}/output.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    # ------------------------------------------------------------------
    # FreqFed sub-procedures (Algorithm 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(weights: List[np.ndarray]) -> np.ndarray:
        """Concatenate all weight arrays into a single 1-D float64 vector."""
        return np.concatenate([w.flatten().astype(np.float64) for w in weights])

    @staticmethod
    def _reshape_to_square(flat: np.ndarray) -> np.ndarray:
        """Reshape a 1-D vector into a square matrix, zero-padding if needed."""
        n = len(flat)
        side = math.ceil(math.sqrt(n))
        padded = np.zeros(side * side, dtype=np.float64)
        padded[:n] = flat
        return padded.reshape(side, side)

    @staticmethod
    def _dct_2d(matrix: np.ndarray) -> np.ndarray:
        """Apply 2-D Type-II DCT with orthonormal normalization (Eq. 2)."""
        return dctn(matrix, type=2, norm='ortho')

    @staticmethod
    def _filter_low_frequencies(V: np.ndarray) -> np.ndarray:
        """
        Extract low-frequency components from DCT coefficient matrix.

        Algorithm 1, lines 13-18 (FILTERING):
            F ← {}
            for i in [0, ⌊|V|/2⌋]:
                for j in [0, ⌊|V|/2⌋]:
                    if i + j <= ⌊|V|/2⌋:
                        F ∪ V[i,j]
            return F

        |V| is the side length of the square matrix.
        """
        side = V.shape[0]
        half = side // 2
        components = []
        for i in range(half + 1):
            for j in range(half + 1):
                if i + j <= half:
                    components.append(V[i, j])
        return np.array(components, dtype=np.float64)

    @staticmethod
    def _cluster(frequency_features: np.ndarray, min_cluster_size: int) -> List[int]:
        """
        Cluster client frequency vectors and return indices of the largest
        cluster.

        Algorithm 1, lines 19-30 (CLUSTERING):
            1. Compute pairwise cosine distance matrix.
            2. Run HDBSCAN with precomputed distances.
            3. Find the cluster label with the most members.
            4. Return indices of clients in that cluster.
        """
        dist_matrix = cosine_distances(frequency_features)

        # Ensure the distance matrix is symmetric and non-negative
        # (numerical precision can produce tiny negatives)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.maximum(dist_matrix, 0.0)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='precomputed',
        )
        labels = clusterer.fit_predict(dist_matrix)

        # Find the largest cluster (excluding noise label -1)
        label_counts = Counter(lbl for lbl in labels if lbl != -1)

        if not label_counts:
            # All points marked as noise — fall back to all clients
            return list(range(len(labels)))

        largest_label = label_counts.most_common(1)[0][0]
        return [idx for idx, lbl in enumerate(labels) if lbl == largest_label]

    # ------------------------------------------------------------------
    # Flower hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        """Build client list for the round. FreqFed is stateless — no
        permanent exclusions between rounds."""
        available_clients = client_manager.all()

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

        # ── Step 1-3: Flatten → square → 2D-DCT → low-freq filter ───────
        client_ids: List[str] = []
        weight_lists: List[List[np.ndarray]] = []
        frequency_features: List[np.ndarray] = []

        for client_proxy, fit_res in results:
            w_list = fl.common.parameters_to_ndarrays(fit_res.parameters)
            client_ids.append(client_proxy.cid)
            weight_lists.append(w_list)

            flat = self._flatten(w_list)
            square = self._reshape_to_square(flat)
            dct_matrix = self._dct_2d(square)
            low_freq = self._filter_low_frequencies(dct_matrix)
            frequency_features.append(low_freq)

        freq_matrix = np.array(frequency_features)  # (K, num_low_freq_components)
        num_clients = len(client_ids)

        # ── Warmup: skip filtering, aggregate all ────────────────────────
        in_warmup = (
            self.begin_removing_from_round is not None
            and self.current_round <= self.begin_removing_from_round
        )

        if in_warmup or not self.remove_clients or num_clients < self.min_cluster_size:
            if in_warmup:
                self.logger.info(
                    f"Round {server_round}: warmup period — aggregating all "
                    f"{num_clients} clients"
                )
            elif num_clients < self.min_cluster_size:
                self.logger.info(
                    f"Round {server_round}: too few clients ({num_clients}) for "
                    f"HDBSCAN — aggregating all"
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

        # ── Steps 4-6: Cluster and select largest cluster ────────────────
        accepted_indices = self._cluster(freq_matrix, self.min_cluster_size)
        accepted_cids = {client_ids[i] for i in accepted_indices}
        self.removed_client_ids = {
            cid for cid in client_ids if cid not in accepted_cids
        }

        time_end = time.time_ns()

        # ── Record metrics ───────────────────────────────────────────────
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start,
            removal_threshold=0.0,
        )
        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=self.removed_client_ids
        )

        for cid in client_ids:
            in_accepted = 1.0 if cid in accepted_cids else 0.0
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=in_accepted,
            )

        # ── Log filtering results ────────────────────────────────────────
        for cid in self.removed_client_ids:
            self.logger.info(
                f"Round {server_round}: filtered out Client {cid}"
            )

        self.logger.info(
            f"Round {server_round}: accepted {len(accepted_cids)}/{num_clients} "
            f"clients | filtered {len(self.removed_client_ids)}"
        )

        # ── Step 7: Simple average over accepted clients (Algorithm 1, line 7)
        #    Gt+1 = (1/L) * Σ Wbl  — equal weight per client, not FedAvg
        accepted_weights = [
            fl.common.parameters_to_ndarrays(res.parameters)
            for proxy, res in results if proxy.cid in accepted_cids
        ]

        if not accepted_weights:
            self.logger.info(
                f"Round {server_round}: no clients accepted — falling back to all"
            )
            return super().aggregate_fit(server_round, results, failures)

        aggregated = [
            np.mean(np.stack([w[layer] for w in accepted_weights]), axis=0)
            for layer in range(len(accepted_weights[0]))
        ]

        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info(
            '\n' + '-' * 50 + f' AGGREGATION ROUND {server_round} ' + '-' * 50
        )

        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics
            accuracy_matrix['cid'] = cid
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=accuracy_matrix['accuracy'],
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
