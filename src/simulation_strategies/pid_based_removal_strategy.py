import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import flwr as fl
from sklearn.cluster import KMeans

from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import (
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    FitRes,
    Parameters,
)
from flwr.server.client_proxy import ClientProxy

from src.output_handlers.directory_handler import DirectoryHandler
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.seed import GLOBAL_SEED

class PIDBasedRemovalStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        ki: float,
        kd: float,
        kp: float,
        num_std_dev: float,
        strategy_history: SimulationStrategyHistory,
        network_model,
        use_lora: bool,
        aggregation_strategy_keyword: str,
        global_seed: int,  # NEW: single source of truth for RNG on server,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client_pids: Dict[str, float] = {}
        self.client_distance_sums: Dict[str, float] = {}
        self.client_distances: Dict[str, float] = {}
        self.current_round: int = 0
        self.removed_client_ids: set[str] = set()

        self.remove_clients = remove_clients
        self.begin_removing_from_round = int(begin_removing_from_round)

        self.ki = float(ki)
        self.kd = float(kd)
        self.kp = float(kp)
        self.num_std_dev = float(num_std_dev)

        self.current_threshold: Optional[float] = None

        self.strategy_history = strategy_history
        self.network_model = network_model
        self.use_lora = bool(use_lora)

        self.aggregation_strategy_keyword = aggregation_strategy_keyword

        # Determinism
        if not global_seed:
            self.global_seed = GLOBAL_SEED
        else:
            self.global_seed = int(global_seed)

        # Logger (avoid duplicate handlers if multiple strategies are constructed)
        self.logger = logging.getLogger(f"pid_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        out_dir = DirectoryHandler.dirname
        os.makedirs(out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{out_dir}/output.log")
        console_handler = logging.StreamHandler()

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def calculate_single_client_pid_scaled(self, client_id, distance):
        """Scale the I component of pid scores."""
        p = distance * self.kp
        if self.current_round == 1:
            return p
        curr_sum = self.client_distance_sums.get(client_id, 0.0)
        i = curr_sum * self.ki
        prev_distance = self.client_distances.get(client_id, 0.0)
        d = self.kd * (distance - prev_distance)
        i_scaled = i / (self.current_round)
        return p + i_scaled + d

    def calculate_single_client_pid(self, client_id, distance):
        """Old PID calculation without scaling or standardization."""
        p = distance * self.kp
        if self.current_round == 1:
            return p
        curr_sum = self.client_distance_sums.get(client_id, 0.0)
        i = curr_sum * self.ki
        prev_distance = self.client_distances.get(client_id, 0.0)
        d = self.kd * (distance - prev_distance)
        return p + i + d

    def calculate_single_client_pid_standardized(self, client_id, distance, avg_sum, sum_std_dev=0.0):
        """Calculate pid with standardized distance."""
        p = distance * self.kp
        if self.current_round == 1:
            return p
        curr_sum = self.client_distance_sums.get(client_id, 0.0)
        i = ((curr_sum - avg_sum) / sum_std_dev) * self.ki if sum_std_dev != 0 else 0.0
        prev_distance = self.client_distances.get(client_id, 0.0)
        d = self.kd * (distance - prev_distance)
        return p + i + d

    def calculate_all_pid_scores(self, results, normalized_distances, standardized=False, scaled=False) -> List[float]:
        pid_scores: List[float] = []

        if self.aggregation_strategy_keyword == "pid_scaled":
            scaled = True
        elif self.aggregation_strategy_keyword == "pid_standardized":
            standardized = True

        if standardized:
            all_sums = sum(self.client_distance_sums.values())
            avg_sum = all_sums / len(self.client_distance_sums) if self.client_distance_sums else 0.0
            sum_dev = np.std(list(self.client_distance_sums.values())) if self.client_distance_sums else 0.0

        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            curr_dist = float(normalized_distances[i][0])
            if standardized:
                new_pid = self.calculate_single_client_pid_standardized(client_id, curr_dist, avg_sum, sum_dev)
            elif scaled:
                new_pid = self.calculate_single_client_pid_scaled(client_id, curr_dist)
            else:
                new_pid = self.calculate_single_client_pid(client_id, curr_dist)
            self.client_pids[client_id] = new_pid
            pid_scores.append(new_pid)

        return pid_scores

    @staticmethod
    def cosine_similarity(tensor1, tensor2):
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
        """Performs aggregation in a determenistic manner for reproducible results."""

        self.current_round = server_round

        # Sort results by cid to remove nondeterminism from return order
        results = sorted(results, key=lambda r: str(r[0].cid))

        # Handle empty results
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        # Filter out removed clients deterministically
        aggregate_clients: List[Tuple[ClientProxy, FitRes]] = [
            r for r in results if r[0].cid not in self.removed_client_ids
        ]

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, aggregate_clients, failures
        )

        # ---- Deterministic clustering & distances ----
        # Flatten parameter tensors for each client in the (sorted) order above
        clustering_param_data = []
        for client_proxy, fit_res in results:
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            params_tensor_list = [torch.tensor(arr) for arr in client_params]
            flattened_param_list = [param.flatten() for param in params_tensor_list]
            param_tensor = torch.cat(flattened_param_list)
            clustering_param_data.append(param_tensor.cpu().numpy())

        X = np.asarray(clustering_param_data, dtype=np.float32)

        # KMeans with fixed random_state & n_init for determinism
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=self.global_seed)
        kmeans.fit(X)
        distances = kmeans.transform(X)  # shape (n_clients, 1)
        normalized_distances = distances  # keep as-is to preserve your semantics
        
        time_start_calc = time.time_ns()
        pids = self.calculate_all_pid_scores(results, normalized_distances)
        time_end_calc = time.time_ns()
        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end_calc - time_start_calc
        )

        counted_pids: List[float] = []
        counted_dist: List[float] = []
        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid
            curr_dist = float(normalized_distances[i][0])
            new_PID = float(pids[i])
            self.client_pids[client_id] = new_PID

            if client_id not in self.removed_client_ids:
                counted_pids.append(new_PID)
                counted_dist.append(curr_dist)

            self.client_distances[client_id] = curr_dist
            curr_sum = self.client_distance_sums.get(client_id, 0.0)
            self.client_distance_sums[client_id] = curr_dist + curr_sum

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_id),
                removal_criterion=new_PID,
                absolute_distance=float(distances[i][0]),
            )

            self.logger.info(
                f"Aggregation round: {server_round} "
                f"Client ID: {client_id} "
                f"PID: {new_PID} "
                f"Normalized Distance: {curr_dist} "
            )

        # Deterministic Threshold computation 
        if self.aggregation_strategy_keyword == "pid":
            pid_avg = float(np.mean(counted_pids)) if counted_pids else 0.0
            pid_std = float(np.std(counted_pids)) if len(counted_pids) > 1 else 0.0
            self.current_threshold = pid_avg + (self.num_std_dev * pid_std) if len(counted_pids) > 1 else 0.0
        else:
            # distance-based threshold for pid_scaled and pid_standardized
            dvals = list(self.client_distances.values())
            distances_avg = float(np.mean(dvals)) if dvals else 0.0
            distances_std = float(np.std(dvals)) if len(dvals) > 1 else 0.0
            self.current_threshold = distances_avg + (self.num_std_dev * distances_std) if len(counted_pids) > 1 else 0.0

        self.strategy_history.insert_round_history_entry(removal_threshold=self.current_threshold)
        self.logger.info(f"REMOVAL THRESHOLD: {self.current_threshold}")

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round, parameters, client_manager):
        # Deterministic view of available clients
        available_clients: Dict[str, ClientProxy] = client_manager.all()
        available_ids = sorted(available_clients.keys(), key=lambda x: str(x))

        # common config info
        cfg_common = {"server_round": int(server_round), "seed": self.global_seed}

        # (Optional) reset removed clients to empty set at the beginning of each round
        # self.removed_client_ids = set()

        # Warmup: select all clients deterministically
        if self.current_round <= self.begin_removing_from_round - 1:
            fit_ins = fl.common.FitIns(parameters, cfg_common)
            return [(available_clients[cid], fit_ins) for cid in available_ids]

        # Deterministic client PID map
        client_pids = {cid: float(self.client_pids.get(cid, 0.0)) for cid in available_ids}

        # Optionally remove clients above threshold
        if self.remove_clients:
            for cid in available_ids:
                pid = client_pids[cid]
                if (pid > float(self.current_threshold)) and (cid not in self.removed_client_ids):
                    self.logger.info(f"Removing client with PID greater than threshold: {cid}")
                    self.removed_client_ids.add(cid)

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )
        self.logger.info(f"removed clients are : {self.removed_client_ids}")

        # Sort by (-PID, cid) for stable tie-breaks
        sorted_client_ids = sorted(
            available_ids,
            key=lambda cid: (-client_pids[cid], int(cid) if str(cid).isdigit() else str(cid)),
        )

        fit_ins = fl.common.FitIns(parameters, cfg_common)
        return [(available_clients[cid], fit_ins) for cid in sorted_client_ids if cid in available_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        # Keep server-side round in sync
        self.current_round = int(server_round)

        # Sort results deterministically
        results = sorted(results, key=lambda r: str(r[0].cid))

        self.logger.info("\n" + "-" * 50 + f"AGGREGATION ROUND {server_round}" + "-" * 50)
        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics
            accuracy_matrix["cid"] = cid

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=accuracy_matrix.get("accuracy"),
            )

        if not results:
            return None, {}

        aggregate_value: List[Tuple[int, float]] = []
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

        for result in results:
            logging.debug(f"Client ID: {result[0].cid}")
            logging.debug(f"Metrics: {result[1].metrics}")
            logging.debug(f"Loss: {result[1].loss}")

        metrics_aggregated: Dict[str, Scalar] = {}

        self.logger.info(
            f"Round: {server_round} "
            f"Number of aggregated clients: {number_of_clients_in_loss_calc} "
            f"Aggregated loss: {loss_aggregated} "
        )

        return loss_aggregated, metrics_aggregated

    # Ensure evaluate() also gets the deterministic config
    def configure_evaluate(self, server_round, parameters, client_manager):
        available_clients: Dict[str, ClientProxy] = client_manager.all()
        available_ids = sorted(available_clients.keys(), key=lambda x: str(x))
        cfg_common = {"server_round": int(server_round), "seed": self.global_seed}
        eval_ins = fl.common.EvaluateIns(parameters, cfg_common)
        return [(available_clients[cid], eval_ins) for cid in available_ids]
