import flwr as fl
import logging

from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import EvaluateRes, Scalar, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class FedAvgStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with round-level metrics tracking.

    This wrapper extends Flower's built-in FedAvg to collect aggregated loss
    and accuracy metrics per round for visualization purposes.
    """

    def __init__(
        self,
        strategy_history: SimulationStrategyHistory,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strategy_history = strategy_history
        self.current_round = 0
        self.logger = logging.getLogger(f"fedavg_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and track round number."""
        self.current_round = server_round
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results and track round-level metrics.

        Collects:
        - Per-client loss and accuracy
        - Aggregated (weighted average) loss
        - Average accuracy across all clients
        """
        if not results:
            return None, {}

        # Collect per-client metrics
        total_examples = 0
        weighted_accuracy_sum = 0.0
        aggregate_loss_values = []

        for client_proxy, evaluate_res in results:
            client_id = int(client_proxy.cid)
            num_examples = evaluate_res.num_examples
            loss = evaluate_res.loss
            accuracy = float(evaluate_res.metrics.get("accuracy", 0.0))

            # Store per-client metrics
            self.strategy_history.insert_single_client_history_entry(
                client_id=client_id,
                current_round=self.current_round,
                loss=loss,
                accuracy=accuracy,
            )

            # Accumulate for aggregation
            aggregate_loss_values.append((num_examples, loss))
            weighted_accuracy_sum += accuracy * num_examples
            total_examples += num_examples

            self.logger.debug(
                f"Round {server_round} - Client {client_id}: "
                f"loss={loss:.4f}, accuracy={accuracy:.4f}, examples={num_examples}"
            )

        # Calculate aggregated metrics
        loss_aggregated = weighted_loss_avg(aggregate_loss_values)
        average_accuracy = (
            weighted_accuracy_sum / total_examples if total_examples > 0 else 0.0
        )

        # Store round-level metrics
        self.strategy_history.rounds_history.aggregated_loss_history.append(
            loss_aggregated
        )
        self.strategy_history.rounds_history.average_accuracy_history.append(
            average_accuracy
        )

        self.logger.info(
            f"Round {server_round}: "
            f"Aggregated loss={loss_aggregated:.4f}, "
            f"Average accuracy={average_accuracy:.4f} "
            f"({len(results)} clients)"
        )

        metrics_aggregated = {"accuracy": average_accuracy}

        return loss_aggregated, metrics_aggregated
