from dataclasses import dataclass

from data_models.simulation_strategy_config import StrategyConfig


@dataclass
class RoundsInfo:

    simulation_strategy_config: StrategyConfig
    score_calculation_time_nanos_history = []
    removal_threshold_history = []

    def add_history_entry(
            self,
            score_calculation_time_nanos: int,
            removal_threshold: float
    ) -> None:
        """
        Add history entry for a new round.

        :param score_calculation_time_nanos: time that a calculation of the score took
        :param removal_threshold: threshold that was used for client removal at current round
        """

        self.score_calculation_time_nanos_history.append(score_calculation_time_nanos)
        self.removal_threshold_history.append(removal_threshold)
