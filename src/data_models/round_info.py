from dataclasses import dataclass, field
from typing import List

from data_models.simulation_strategy_config import StrategyConfig


@dataclass
class RoundsInfo:

    simulation_strategy_config: StrategyConfig

    score_calculation_time_nanos_history: List[int] = field(default_factory=list)
    removal_threshold_history: List[float] = field(default_factory=list)
    aggregated_loss_history: List[float] = field(default_factory=list)
    average_accuracy_history: List[float] = field(default_factory=list)

    tp_history: List[float] = field(default_factory=list)
    tn_history: List[float] = field(default_factory=list)
    fp_history: List[float] = field(default_factory=list)
    fn_history: List[float] = field(default_factory=list)
    total_fp_and_fn_history: List[float] = field(default_factory=list)

    removal_accuracy_history: List[float] = field(default_factory=list)
    removal_precision_history: List[float] = field(default_factory=list)
    removal_recall_history: List[float] = field(default_factory=list)
    removal_f1_history: List[float] = field(default_factory=list)

    plottable_metrics = [
        "score_calculation_time_nanos_history",
        "aggregated_loss_history",
        "average_accuracy_history",
        "removal_accuracy_history",
        "removal_precision_history",
        "removal_recall_history",
        "removal_f1_history",
        "total_fp_and_fn_history",
    ]

    savable_metrics = [
        "score_calculation_time_nanos_history",
        "removal_threshold_history",
        "aggregated_loss_history",
        "average_accuracy_history",
        "tp_history",
        "tn_history",
        "fp_history",
        "fn_history",
        "removal_accuracy_history",
        "removal_precision_history",
        "removal_recall_history",
        "removal_f1_history",
    ]

    def add_history_entry(
            self,
            score_calculation_time_nanos: int,
            removal_threshold: float,
            aggregated_loss: float,
            average_accuracy: float
    ) -> None:
        """
        Add history entry for a new round.

        :param score_calculation_time_nanos: time that a calculation of the score took
        :param removal_threshold: threshold that was used for client removal at current round
        :param aggregated_loss: loss that was aggregated among participating clients in this round
        :param average_accuracy:
        """

        self.score_calculation_time_nanos_history.append(score_calculation_time_nanos)
        self.removal_threshold_history.append(removal_threshold)
        self.aggregated_loss_history.append(aggregated_loss)
        self.average_accuracy_history.append(average_accuracy)

    def get_metric_by_name(self, metric: str) -> List:
        """Get single plottable metric values by name"""

        return getattr(self, metric)

    def append_tp_tn_fp_fn(self, round_tp, round_tn, round_fp, round_fn) -> None:
        """Append round metrics"""

        self.tp_history.append(round_tp)
        self.tn_history.append(round_tn)
        self.fp_history.append(round_fp)
        self.fn_history.append(round_fn)

    def calculate_additional_metrics(self) -> None:
        """
        Calculate the following metrics by iterating over tp, tn, fp, fn:
            accuracy, precision, recall, f1-score
        """

        for round_tp, round_tn, round_fp, round_fn in zip(
                self.tp_history, self.tn_history, self.fp_history, self.fn_history
        ):
            # accuracy: (tp + tn) / (tp + tn + fp + fn)
            self.removal_accuracy_history.append(
                (round_tp + round_tn) / (round_tp + round_tn + round_fp + round_fn)
            )

            # precision: tp / (tp + fp)
            precision = round_tp / (round_tp + round_fp)
            self.removal_precision_history.append(precision)

            # recall: tp / (tp + fn)
            recall = round_tp / (round_tp + round_fn)
            self.removal_recall_history.append(recall)

            # f1: 2 * precision * recall / (precision + recall)
            self.removal_f1_history.append(
                2 * precision * recall / (precision + recall)
            )

            self.total_fp_and_fn_history.append(round_fp + round_fn)
