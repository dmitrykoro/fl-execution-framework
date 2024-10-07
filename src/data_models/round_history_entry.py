from dataclasses import dataclass


@dataclass
class RoundHistoryEntry:
    """Represents possible metrics collected after the execution of aggregation round"""

    loss: float
    accuracy: float
    removal_criterion: float
    absolute_distance: float
    normalized_distance: float
    is_removed: bool

    def __init__(
            self,
            loss: float = None,
            accuracy: float = None,
            removal_criterion: float = None,
            absolute_distance: float = None,
            normalized_distance: float = None,
            is_removed: bool = None
    ) -> None:
        self.loss = float(loss)
        self.accuracy = float(accuracy)
        self.removal_criterion = float(removal_criterion)
        self.absolute_distance = float(absolute_distance)
        self.normalized_distance = float(normalized_distance)
        self.is_removed = bool(is_removed)
