import time
import numpy as np
import flwr as fl
import torch
import inspect
import logging
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

class RFABasedRemovalStrategy(FedAvg):
    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        weighted_median_factor: float = 1.0,
        strategy_history=None,
        num_of_malicious_clients: int = 0,
        *args,
        **kwargs,
    ):
        fedavg_params = set(inspect.signature(FedAvg.__init__).parameters) - {"self"}
        fedavg_kwargs = {k: v for k, v in kwargs.items() if k in fedavg_params}
        super().__init__(*args, **fedavg_kwargs)
        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round
        self.weighted_median_factor = weighted_median_factor
        self.current_round = 0
        self.removed_client_ids = set()
        self.client_scores = {}
        self.strategy_history = strategy_history
        self.num_of_malicious_clients = num_of_malicious_clients
        self.rounds_history = {}

    def _geometric_median(self, stacked_params: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        guess = np.mean(stacked_params, axis=0)
        while True:
            diff = stacked_params - guess
            dist = np.linalg.norm(diff, axis=1)
            dist = np.where(dist < eps, eps, dist)
            weights = 1.0 / dist
            new_guess = np.average(stacked_params, axis=0, weights=weights)
            if np.linalg.norm(new_guess - guess) < eps:
                return new_guess
            guess = new_guess


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return None, {}

        clustering_param_data = []
        for _, fit_res in results:
            tensors = [torch.tensor(arr).flatten() for arr in fl.common.parameters_to_ndarrays(fit_res.parameters)]
            clustering_param_data.append(torch.cat(tensors))
        X_embed = np.vstack([t.numpy() for t in clustering_param_data])
        kmeans = KMeans(n_clusters=1, init="k-means++").fit(X_embed)
        abs_distances = kmeans.transform(X_embed)
        norm_distances = MinMaxScaler().fit(abs_distances).transform(abs_distances)

        param_arrays = [fl.common.parameters_to_ndarrays(fr.parameters) for _, fr in results]
        flat_updates = np.stack([np.concatenate([p.ravel() for p in pa]) for pa in param_arrays])

        time_start_calc = time.time_ns()
        geometric_median = self._geometric_median(flat_updates)
        weighted_geometric_median = geometric_median * self.weighted_median_factor
        time_end_calc = time.time_ns()

        agg_list, cursor = [], 0
        for arr in param_arrays[0]:
            num = arr.size
            agg_list.append(
                weighted_geometric_median[cursor : cursor + num].reshape(arr.shape).astype(arr.dtype)
            )
            cursor += num
        aggregated_parameters = fl.common.ndarrays_to_parameters(agg_list)

        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end_calc - time_start_calc
        )

        for i, (client_proxy, _) in enumerate(results):
            cid = client_proxy.cid
            deviation = float(np.linalg.norm(flat_updates[i] - weighted_geometric_median))
            self.client_scores[cid] = deviation
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=deviation,
                absolute_distance=float(abs_distances[i][0]),
            )
            logging.info(
                f"Aggregation round: {server_round} Client ID: {cid} Deviation: {deviation:.4e} "
                f"Normalized Distance: {norm_distances[i][0]:.4f}"
            )

        return aggregated_parameters, {}


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        logging.info("\n" + "-" * 50 + f"AGGREGATION ROUND {server_round}" + "-" * 50)

        for cp, ev in results:
            cid = cp.cid
            acc_metrics = dict(ev.metrics)
            acc_metrics["cid"] = cid
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=acc_metrics.get("accuracy"),
            )

        if not results:
            return None, {}

        aggregate_value, num_clients_loss = [], 0
        for cp, ev in results:
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cp.cid), current_round=self.current_round, loss=ev.loss
            )
            if cp.cid not in self.removed_client_ids:
                aggregate_value.append((ev.num_examples, ev.loss))
                num_clients_loss += 1

        loss_aggregated = weighted_loss_avg(aggregate_value) if aggregate_value else float("nan")
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        for cp, ev in results:
            logging.debug(f"Client ID: {cp.cid} Metrics: {ev.metrics} Loss: {ev.loss}")

        logging.info(
            f"Round: {server_round} Number of aggregated clients: {num_clients_loss} "
            f"Aggregated loss: {loss_aggregated}"
        )
        return loss_aggregated, {}


