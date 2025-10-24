import numpy as np

from dataclasses import dataclass, field
from typing import Optional

from src.data_models.client_info import ClientInfo
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig

from src.dataset_handlers.dataset_handler import DatasetHandler
from src.attack_utils.poisoning import should_poison_this_round


@dataclass
class SimulationStrategyHistory:

    strategy_config: StrategyConfig
    dataset_handler: DatasetHandler
    rounds_history: Optional[RoundsInfo] = None
    _clients_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rounds_history = RoundsInfo(simulation_strategy_config=self.strategy_config)

        for i in range(self.strategy_config.num_of_clients):
            self._clients_dict[i] = ClientInfo(
                client_id=i,
                num_of_rounds=self.strategy_config.num_of_rounds,
                is_malicious=False
            )

    def get_all_clients(self) -> list:
        """Get list of all ClientInfo instances"""

        return [client for client in self._clients_dict.values()]

    def insert_single_client_history_entry(
            self,
            client_id: int,
            current_round: int,
            removal_criterion: float = None,
            absolute_distance: float = None,
            loss: float = None,
            accuracy: float = None,
            aggregation_participation: int = None
    ) -> None:
        """Insert history entry for a single client. Only those values provided will be updated."""

        updating_client: ClientInfo = self._clients_dict[client_id]

        updating_client.add_history_entry(
            current_round,
            removal_criterion,
            absolute_distance,
            loss,
            accuracy,
            aggregation_participation
        )

    def insert_round_history_entry(
            self,
            score_calculation_time_nanos: int = None,
            removal_threshold: float = None,
            loss_aggregated: float = None
    ) -> None:
        """Append the round history info to the history. Only those values provided will be updated."""

        if score_calculation_time_nanos:
            self.rounds_history.score_calculation_time_nanos_history.append(score_calculation_time_nanos)
        if removal_threshold:
            self.rounds_history.removal_threshold_history.append(removal_threshold)
        if loss_aggregated:
            self.rounds_history.aggregated_loss_history.append(loss_aggregated)

    def update_client_participation(
            self,
            current_round: int,
            removed_client_ids: set
    ) -> None:
        """Update history of client participation based on the IDs of removed clients at the given round."""

        for client_id in removed_client_ids:
            self.insert_single_client_history_entry(
                client_id=int(client_id),
                current_round=current_round,
                aggregation_participation=0
            )

    def update_client_malicious_status(self, current_round: int) -> None:
        """
        Update client.is_malicious flag based on attack_schedule for the current round.

        Args:
            current_round: Current training round (1-indexed)
        """
        # Only update if attack_schedule is configured
        if not self.strategy_config.attack_schedule:
            return

        # Update each client's malicious status based on current round
        for client_id in range(self.strategy_config.num_of_clients):
            should_poison, _ = should_poison_this_round(
                current_round=current_round,
                client_id=client_id,
                attack_schedule=self.strategy_config.attack_schedule
            )

            # Update the is_malicious flag
            self._clients_dict[client_id].is_malicious = should_poison

    def calculate_additional_rounds_data(self) -> None:
        """
        The primary data that is collected during the simulation is as follows:

        Per-client data ver rounds (stored in ClientInfo):
            loss_history,
            accuracy_history - achieved accuracy during the validation,
            removal_criterion_history - the calculated value based on which the removal is performed,
            absolute_distance_history (to cluster center) - the absolute distance to the center of the cluster of all models

        Per-round data  over rounds (stored in RoundsInfo):
            aggregated_loss_history - loss of all aggregated client models
            score_calculation_time_nanos_history - time that was used to calculate removal_criterion
            removal_threshold_history - removal threshold that was used for exclusion at each round

        The following derivative data is calculated here (for each round):

        average_accuracy_history - average accuracy of all benign clients that are not removed at a given round

        <to be continued>

        """

        for round_num in range(self.strategy_config.num_of_rounds):

            round_tp_count = 0
            round_tn_count = 0
            round_fp_count = 0
            round_fn_count = 0

            num_aggregated_clients = 0
            sum_aggregated_accuracies = 0

            round_client_accuracies = []

            for client_info in self.get_all_clients():

                client_is_malicious = client_info.is_malicious
                client_was_aggregated = client_info.aggregation_participation_history[round_num] == 1

                if self.strategy_config.remove_clients:
                    # true positive: a good client was aggregated
                    if not client_is_malicious and client_was_aggregated:
                        round_tp_count += 1
                    # true negative: a malicious client was not aggregated
                    if client_is_malicious and not client_was_aggregated:
                        round_tn_count += 1
                    # false positive: a good client was not aggregated
                    if not client_is_malicious and not client_was_aggregated:
                        round_fp_count += 1
                    # false negative: a malicious client was aggregated
                    if client_is_malicious and client_was_aggregated:
                        round_fn_count += 1

                # sum of accuracies of aggregated clients
                if client_was_aggregated:
                    num_aggregated_clients += 1
                    sum_aggregated_accuracies += client_info.accuracy_history[round_num]
                    round_client_accuracies.append(client_info.accuracy_history[round_num])

            self.rounds_history.append_tp_tn_fp_fn(round_tp_count, round_tn_count, round_fp_count, round_fn_count)
            self.rounds_history.average_accuracy_history.append(
                float(f"{(sum_aggregated_accuracies / num_aggregated_clients * 100):.2f}") if num_aggregated_clients > 0 else 0.000
            )
            self.rounds_history.average_accuracy_std_history.append(float( f"{np.std(round_client_accuracies) * 100:.2f}"))

        if self.strategy_config.remove_clients:
            self.rounds_history.calculate_additional_metrics()
