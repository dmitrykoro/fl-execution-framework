from dataclasses import dataclass


@dataclass
class ClientInfo:

    id: int
    is_malicious: bool = None
    removal_criterion_history = []
    absolute_distance_history = []
    loss_history = []
    accuracy_history = []
    aggregation_participation_history = []

    def add_history_entry(
            self,
            removal_criterion: float = None,
            absolute_distance: float = None,
            loss: float = None,
            accuracy: float = None,
            aggregation_participation: int = None
    ) -> None:
        """
        Add client metrics history entry for a new round.

        :param accuracy: accuracy at this round
        :param loss: loss at this round
        :param removal_criterion: calculated score based on which the removal was or was not performed
        :param absolute_distance: to the centroid of all models
        :param aggregation_participation: 1 if client was aggregated, 0 if was not
        """

        if removal_criterion:
            self.removal_criterion_history.append(removal_criterion)
        if absolute_distance:
            self.absolute_distance_history.append(absolute_distance)
        if loss:
            self.loss_history.append(loss)
        if accuracy:
            self.accuracy_history.append(accuracy)
        if aggregation_participation:
            self.aggregation_participation_history.append(aggregation_participation)
