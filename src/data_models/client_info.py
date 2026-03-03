from dataclasses import dataclass
from typing import List, Union, Optional


@dataclass
class ClientInfo:

    client_id: int
    num_of_rounds: int
    is_malicious: bool = None
    rounds: List[int] = None

    removal_criterion_history: List[Union[float, None]] = None
    absolute_distance_history: List[Union[float, None]] = None
    loss_history: List[Union[float, None]] = None
    accuracy_history: List[Union[float, None]] = None
    ss_res_history: List[Union[float, None]] = None
    ss_tot_history: List[Union[float, None]] = None
    r2_history: List[Union[float, None]] = None
    aggregation_participation_history: List[Union[int, None]] = None

    plottable_metrics = [
        "removal_criterion_history",
        "absolute_distance_history",
        "loss_history",
        "accuracy_history",
        "r2_history"
    ]

    savable_metrics = [
        "removal_criterion_history",
        "absolute_distance_history",
        "loss_history",
        "accuracy_history",
        "r2_history",
        "aggregation_participation_history",
    ]

    def __post_init__(self):
        def _init_list(value=None):
            return [value] * self.num_of_rounds

        self.removal_criterion_history = _init_list()
        self.absolute_distance_history = _init_list()
        self.loss_history = _init_list()
        self.accuracy_history = _init_list()
        self.ss_res_history = _init_list()
        self.ss_tot_history = _init_list()
        self.r2_history = _init_list()
        self.aggregation_participation_history = _init_list(value=1)

        self.rounds = []

        for round_num in range(self.num_of_rounds):
            self.rounds.append(round_num + 1)

    def add_history_entry(
            self,
            current_round: int,
            removal_criterion: Optional[float] = None,
            absolute_distance: Optional[float] = None,
            loss: Optional[float] = None,
            accuracy: Optional[float] = None,
            ss_res: Optional[float] = None,
            ss_tot: Optional[float] = None,
            r2: Optional[float] = None,
            aggregation_participation: Optional[int] = None
    ) -> None:
        """
        Add client metrics history entry for a new round.

        :param current_round: specifies the round number to put all data into correct index at list
        :param accuracy: accuracy at this round
        :param loss: loss at this round
        :param ss_res: residual sum of squares between ground truth and predicted used to calculate global R^2
        :param ss_tot: total sum of squares between ground truth and mean used to calculate global R^2
        :param r2: r2 at this round
        :param removal_criterion: calculated score based on which the removal was or was not performed
        :param absolute_distance: to the centroid of all models
        :param aggregation_participation: 1 if client was aggregated, 0 if was not
        """

        if removal_criterion is not None:
            self.removal_criterion_history[current_round - 1] = removal_criterion
        if absolute_distance is not None:
            self.absolute_distance_history[current_round - 1] = absolute_distance
        if loss is not None:
            self.loss_history[current_round - 1] = loss
        if accuracy is not None:
            self.accuracy_history[current_round - 1] = accuracy
        if ss_res is not None:
            self.ss_res_history[current_round - 1] = ss_res
        if ss_tot is not None:
            self.ss_tot_history[current_round - 1] = ss_tot
        if r2 is not None:
            self.r2_history[current_round - 1] = r2
        if aggregation_participation is not None:
            self.aggregation_participation_history[current_round - 1] = aggregation_participation

    def get_metric_by_name(self, metric: str) -> List:
        """Get single plottable or savable metric values by name"""

        return getattr(self, metric)
