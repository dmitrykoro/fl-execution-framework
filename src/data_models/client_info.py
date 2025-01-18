from dataclasses import dataclass
from typing import List


@dataclass
class ClientInfo:

    client_id: int
    num_of_rounds: int
    is_malicious: bool = None
    removal_criterion_history: List[float or None] = None
    absolute_distance_history: List[float or None] = None
    loss_history: List[float or None] = None
    accuracy_history: List[float or None] = None
    aggregation_participation_history: List[int or None] = None

    def __post_init__(self):
        def _init_list():
            return [None] * self.num_of_rounds

        self.removal_criterion_history = _init_list()
        self.absolute_distance_history = _init_list()
        self.loss_history = _init_list()
        self.accuracy_history = _init_list()
        self.aggregation_participation_history = _init_list()

    def add_history_entry(
            self,
            current_round: int,
            removal_criterion: float = None,
            absolute_distance: float = None,
            loss: float = None,
            accuracy: float = None,
            aggregation_participation: int = None
    ) -> None:
        """
        Add client metrics history entry for a new round.

        :param current_round: specifies the round number to put all data into correct index at list
        :param accuracy: accuracy at this round
        :param loss: loss at this round
        :param removal_criterion: calculated score based on which the removal was or was not performed
        :param absolute_distance: to the centroid of all models
        :param aggregation_participation: 1 if client was aggregated, 0 if was not
        """

        if removal_criterion:
            self.removal_criterion_history[current_round - 1] = removal_criterion
        if absolute_distance:
            self.absolute_distance_history[current_round - 1] = absolute_distance
        if loss:
            self.loss_history[current_round - 1] = loss
        if accuracy:
            self.accuracy_history[current_round - 1] = accuracy
        if aggregation_participation:
            self.aggregation_participation_history[current_round - 1] = aggregation_participation
