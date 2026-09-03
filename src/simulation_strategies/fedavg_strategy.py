import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.utils.medmentions_strategy_helper import aggregate_strict_medmentions_metrics
from src.utils.strategy_evaluation_metrics_helper import store_evaluation_metrics


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Vanilla FedAvg wrapped as a first-class InteFL strategy.

    Adds:
    - score_calculation_time_nanos: time to flatten all client weight vectors
      and compute their mean — the baseline O(N·d) cost shared by every
      distance-based strategy. Useful for overhead comparison.
    - Benign-only loss/accuracy aggregation consistent with all other strategies.
    - remove_clients is always forced to False; passing True is a no-op.
    """

    def __init__(
        self,
        remove_clients: bool,
        begin_removing_from_round: int,
        strategy_history: SimulationStrategyHistory,
        output_dirname: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.strategy_history = strategy_history
        self.current_round: int = 0

        self.logger = logging.getLogger(f"fedavg_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)
        os.makedirs(output_dirname, exist_ok=True)
        self.logger.addHandler(logging.FileHandler(f"{output_dirname}/output.log"))
        self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False

    def configure_fit(self, server_round, parameters, client_manager):
        available_clients = client_manager.all()
        self.strategy_history.update_client_participation(
            current_round=server_round, removed_client_ids=set()
        )
        fit_ins = fl.common.FitIns(parameters, {"server_round": server_round})
        return [(client, fit_ins) for client in available_clients.values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        # Time the flattening + mean: the baseline cost every distance-based
        # strategy also pays, so this gives a fair comparison floor.
        time_start = time.time_ns()
        flat_weights = []
        for _, fit_res in results:
            wl = parameters_to_ndarrays(fit_res.parameters)
            flat_weights.append(
                np.concatenate([w.flatten().astype(np.float64) for w in wl])
            )
        _ = np.mean(flat_weights, axis=0)
        time_end = time.time_ns()

        self.strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=time_end - time_start
        )

        for i, (client_proxy, _) in enumerate(results):
            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(client_proxy.cid),
                absolute_distance=float(np.linalg.norm(
                    flat_weights[i] - np.mean(flat_weights, axis=0)
                )),
            )

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        self.logger.info(
            "\n" + "-" * 50 + f" AGGREGATION ROUND {server_round} " + "-" * 50
        )

        for client_result in results:
            cid = client_result[0].cid
            metrics = client_result[1].metrics
            metrics["cid"] = cid
            # Dispatch on which keys the payload carries: flower_client.evaluate
            # returns accuracy for classification, {ss_res, ss_tot, r2} for
            # regression and {loss_baseline, mare, ...} for time-series models.
            store_evaluation_metrics(
                clinet_id=int(cid),
                current_round=self.current_round,
                metrics=metrics,
                strategy_history=self.strategy_history,
            )

        if not results:
            return None, {}

        malicious_ids = self.strategy_history.get_malicious_client_ids()
        aggregate_value = []

        for client_metadata, evaluate_res in results:
            cid = int(client_metadata.cid)
            self.strategy_history.insert_single_client_history_entry(
                client_id=cid,
                current_round=self.current_round,
                loss=evaluate_res.loss,
            )
            if cid not in malicious_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = aggregate_strict_medmentions_metrics(
            results=results,
            removed_client_ids=set(),
            remove_clients=False,
            current_round=self.current_round,
            begin_removing_from_round=None,
            strategy_history=self.strategy_history,
        )

        self.logger.info(
            f"Round: {server_round} | Loss: {loss_aggregated:.4f}"
        )

        return loss_aggregated, metrics_aggregated
