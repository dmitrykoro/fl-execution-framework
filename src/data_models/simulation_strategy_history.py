from dataclasses import dataclass, field

from data_models.client_info import ClientInfo
from data_models.round_info import RoundsInfo
from data_models.simulation_strategy_config import StrategyConfig


@dataclass
class SimulationStrategyHistory:

    strategy_config: StrategyConfig
    _clients_dict: dict = field(default_factory=dict)
    rounds_history: RoundsInfo = field(init=False)

    def __post_init__(self):
        self.rounds_history = RoundsInfo(simulation_strategy_config=self.strategy_config)

        for i in range(self.strategy_config.num_of_clients):
            self._clients_dict[i] = ClientInfo(client_id=i, num_of_rounds=self.strategy_config.num_of_rounds)

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
        updating_client = self._clients_dict[client_id]

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
            removal_threshold: float = None
    ) -> None:

        if score_calculation_time_nanos:
            self.rounds_history.score_calculation_time_nanos_history.append(score_calculation_time_nanos)
        if removal_threshold:
            self.rounds_history.removal_threshold_history.append(removal_threshold)
