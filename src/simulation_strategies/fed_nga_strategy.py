import time
from flwr.server.client_manager import ClientManager
import numpy as np
import flwr as fl
import torch
import logging
from sklearn.cluster import KMeans
import os

from typing import Dict, List, Optional, Tuple, Union, Set

from flwr.common import FitIns, FitRes, MetricsAggregationFn, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.output_handlers.directory_handler import DirectoryHandler



class FedNGAStrategy(FedAvg):
    def __init__(
        self, 
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        output_dirname,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.current_round = 0
        self.removed_client_ids = set()
        self.client_scores = {}

        self.strategy_history = strategy_history
        self.current_parameters = None

        # Create a logger
        self.logger = logging.getLogger(f"fed_nga_{id(self)}")
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


    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Tuple[ClientProxy, FitRes] | BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Increment the current round counter.
        self.current_round += 1

        if not self.current_parameters:
            return super().aggregate_fit(server_round, results, failures)
        
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

        """
        Receiving update parameters from clients.
        However, Fed-NGA needs client gradients.
        So need to calcualte the gradients from the parameters
        w_c = w_g - (η_local · g) => g = (w_g - w_c) / η_local
        """
        # Current global model
        w_global = parameters_to_ndarrays(self.current_parameters)

        # Compute client deltas
        # given parameter updates, need to change them back to client gradients
        deltas = []
        num_examples = []
        for i, (client, fit_res) in enumerate(results):
            w_client = parameters_to_ndarrays(fit_res.parameters)
            eta_eff = fit_res.metrics["eta_eff"]
            delta = [(wg - wc)/eta_eff for wg, wc in zip(w_global, w_client)]
            deltas.append(delta)

            num_examples.append(fit_res.num_examples)

            self.client_scores[int(client.cid)] = 0.0
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client.cid),
                current_round=self.current_round,
                removal_criterion=0.0,
                absolute_distance=float(distances[i][0])
            )

        """
        After obtaining client gradients, can perform Fed-NGA aggregation
        w^(t+1) = w^t - η^t∑_m∈M (a_m · (g_m^t / ||g_m^t||))
        """
        time_start_calc = time.time_ns()
        
        # Normalize deltas (Fed-NGA core)
        # g_m^t / ||g_m^t||
        normalized_deltas = []
        for delta in deltas:
            norm = np.sqrt(sum(np.sum(layer ** 2) for layer in delta))
            if norm > 0:
                normalized_deltas.append([layer / norm for layer in delta])
            else:
                normalized_deltas.append(delta)

        # weighted aggregation
        # g^t = ∑_m∈M (a_m · (g_m^t / ||g_m^t||))
        total = sum(num_examples)
        weights = [n/total for n in num_examples]

        avg_delta = []
        for layers in zip(*normalized_deltas):
            avg_delta.append(sum(w * layer for w, layer in zip (weights, layers)))

        # Server-side learning rate per round
        # η_t = η / (t+c)^(1/2 + δ)
        g_norm = np.sqrt(sum(np.sum(layer ** 2) for layer in avg_delta))

        # Adaptive eta0 based on consensus strength
        eta0_min = 1.0
        eta0_max = 5.0
        tau = 1.0 # controls sensitivity

        eta0_dyn = eta0_min + (eta0_max - eta0_min) * (1 - np.exp(-g_norm / tau))

        fine_tune = 0.1
        c = 100
        eta_t = eta0_dyn / ((server_round + c)**(0.5+fine_tune))

        # new global model
        # w^(t+1) = w^t - η^t · g^t
        w_new = [
            wg - eta_t * d
            for wg, d in zip(w_global, avg_delta)
        ]

        time_end_calc = time.time_ns()
        self.strategy_history.insert_round_history_entry(score_calculation_time_nanos=time_end_calc - time_start_calc)

        for i, (client_proxy, _) in enumerate(results):
            client_id = client_proxy.cid

            self.logger.info(
                f'Aggregation round: {server_round} '
                f'Client ID: {client_id} '
                f'Score: {0.0} '
                f'Absolute Distance: {distances[i][0]}'
            )

        self.logger.info(
            f"Round {server_round}: "
                f"eta={eta_t:.6f}, "
                f"||g||={np.sqrt(sum(np.sum(l**2) for l in avg_delta)):.3f}"
        )

        #Update strategy history
        self.strategy_history.update_client_participation(
            current_round=self.current_round,
            removed_client_ids=set()
        )

        new_params = ndarrays_to_parameters(w_new)
        self.current_parameters = new_params
        return new_params, {}


    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        
        self.current_parameters = parameters
        
        # fetch available clients as a dictionary
        available_clients = client_manager.all()    # dictionary with client IDs as keys and RayActorClientProxy objects as values

        client_scores = {client_id: self.client_scores.get(client_id, 0) for client_id in available_clients.keys()}

        selected_client_ids = sorted(client_scores, key=client_scores.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {})

        return [(available_clients[cid], fit_ins) for cid in selected_client_ids if cid in available_clients]
    

    def aggregate_evaluate(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, EvaluateRes]], 
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        self.logger.info('\n' + '-' * 50 + f'AGGREGATION ROUND {server_round}' + '-' * 50)

        if not results:
            return super().aggregate_evaluate(server_round, results, failures)
        
        for client_result in results:
            cid = client_result[0].cid
            metrics = client_result[1].metrics
            metrics['cid'] = cid

            if "accuracy" in metrics:
                # For classification tasks
                self.strategy_history.insert_single_client_history_entry(
                    client_id=int(cid),
                    current_round=self.current_round,
                    accuracy=metrics['accuracy']
                )
            elif "ss_res" in metrics and "ss_tot" in metrics and "r2" in metrics:
                # For regression tasks
                self.strategy_history.insert_single_client_history_entry(
                    client_id=int(cid),
                    current_round=self.current_round,
                    ss_res=metrics["ss_res"],
                    ss_tot=metrics["ss_tot"],
                    r2=metrics["r2"]
                )

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
        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=self.removed_client_ids,
            remove_clients=self.remove_clients,
            current_round=self.current_round,
            begin_removing_from_round=self.begin_removing_from_round,
            strategy_history=self.strategy_history
        )

        self.logger.info(
            f'Round: {server_round} '
            f'Number of aggregated clients: {number_of_clients_in_loss_calc} '
            f'Aggregated loss: {loss_aggregated} '
        )

        return loss_aggregated, metrics_aggregated


