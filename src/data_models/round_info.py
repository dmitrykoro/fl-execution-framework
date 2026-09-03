from dataclasses import dataclass, field
from typing import List, Any
from numbers import Number
import numpy as np

from src.data_models.simulation_strategy_config import StrategyConfig


@dataclass
class RoundsInfo:
    simulation_strategy_config: StrategyConfig

    score_calculation_time_nanos_history: List[int] = field(default_factory=list)
    removal_threshold_history: List[float] = field(default_factory=list)
    aggregated_loss_history: List[float] = field(default_factory=list)
    average_accuracy_history: List[float] = field(default_factory=list)
    average_accuracy_std_history: List[float] = field(default_factory=list)

    tp_history: List[float] = field(default_factory=list)
    tn_history: List[float] = field(default_factory=list)
    fp_history: List[float] = field(default_factory=list)
    fn_history: List[float] = field(default_factory=list)
    total_fp_and_fn_history: List[float] = field(default_factory=list)

    removal_accuracy_history: List[float] = field(default_factory=list)
    removal_precision_history: List[float] = field(default_factory=list)
    removal_recall_history: List[float] = field(default_factory=list)
    removal_f1_history: List[float] = field(default_factory=list)
    mention_precision_history: List[float] = field(default_factory=list)
    mention_recall_history: List[float] = field(default_factory=list)
    mention_f1_history: List[float] = field(default_factory=list)
    document_precision_history: List[float] = field(default_factory=list)
    document_recall_history: List[float] = field(default_factory=list)
    document_f1_history: List[float] = field(default_factory=list)

    # Aggregate metrics for Regression task datasets (e.g. air_quality_uci)
    average_r2_history: List[float] = field(default_factory=list) # Aggregated R^2 Value across all clients

    # Aggregate metrics for Time-series forcasting datases (e.g. smores)
    average_loss_baseline_history: List[float] = field(default_factory=list)
    average_mare_history: List[float] = field(default_factory=list)
    average_mare_baseline_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.plottable_metrics = [
            "score_calculation_time_nanos_history",
            "aggregated_loss_history",
        ]
        self.time_series_plottable_metrics = []
        if self.simulation_strategy_config.model_type in ["cnn", "transformer"]:
            self.plottable_metrics += ["average_accuracy_history"]
        elif self.simulation_strategy_config.model_type in ["linear_regression", "mlp"]:
            self.plottable_metrics += ["average_r2_history"]
        elif self.simulation_strategy_config.model_type == "lstm":
            self.plottable_metrics += ["average_mare_history"]
            self.time_series_plottable_metrics += ["aggregated_loss_history", "average_loss_baseline_history", "average_mare_history", "average_mare_baseline_history"]

        self.plottable_metrics += [
            "removal_accuracy_history",
            "removal_precision_history",
            "removal_recall_history",
            "removal_f1_history",
        ]

        self.barable_metrics = [
            "total_fp_and_fn_history",
        ]

        self.savable_metrics = [
            "score_calculation_time_nanos_history",
            "removal_threshold_history",
            "aggregated_loss_history",
        ]
        if self.simulation_strategy_config.model_type in ["cnn", "transformer"]:
            self.savable_metrics += ["average_accuracy_history", "average_accuracy_std_history"]
        elif self.simulation_strategy_config.model_type in ["linear_regression", "mlp"]:
            self.savable_metrics += ["average_r2_history"]
        elif self.simulation_strategy_config.model_type == "lstm":
            self.savable_metrics += ["average_loss_baseline_history", "average_mare_history", "average_mare_baseline_history"]

        self.statsable_metrics = [
            "tp_history",
            "tn_history",
            "fp_history",
            "fn_history",
            "removal_accuracy_history",
            "removal_precision_history",
            "removal_recall_history",
            "removal_f1_history",
        ]
        if self.simulation_strategy_config.model_type in ["cnn", "transformer"]:
            self.statsable_metrics = ["average_accuracy_history",] + self.statsable_metrics
        
        if self.simulation_strategy_config.dataset_keyword == "medmentions":
            self.savable_metrics += [
                "document_precision_history",
                "document_recall_history",
                "document_f1_history",
            ]
            self.plottable_metrics += [
                "document_precision_history",
                "document_recall_history",
                "document_f1_history",
            ]

        if self.simulation_strategy_config.remove_clients:
            self.savable_metrics += self.statsable_metrics

    @staticmethod
    def _to_python_number(value: Any) -> Any:
        """
        Convert value to python number to allow for later database upload.
        Catches np.float32, np.float64, etc. which are not json serializable.
        """
        if isinstance(value, np.generic):
            return value.item()
        return value 

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
        self.removal_threshold_history.append(self._to_python_number(removal_threshold))
        self.aggregated_loss_history.append(self._to_python_number(aggregated_loss))
        self.average_accuracy_history.append(self._to_python_number(average_accuracy))

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
            try:
                precision = round_tp / (round_tp + round_fp)
                self.removal_precision_history.append(precision)
            except ZeroDivisionError:
                self.removal_precision_history.append(
                    f"impossible to calculate precision tp / (tp + fp): {round_tp} / ({round_tp + round_fp})"
                )

            # recall: tp / (tp + fn)
            try:
                recall = round_tp / (round_tp + round_fn)
                self.removal_recall_history.append(recall)
            except ZeroDivisionError:
                self.removal_recall_history.append(
                    float("inf")
                )
        
            

            # f1: 2 * precision * recall / (precision + recall)
            try:
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                self.removal_f1_history.append(f1)
            except UnboundLocalError:
                self.removal_f1_history.append("Impossible to calculate")

            self.total_fp_and_fn_history.append(round_fp + round_fn)

    def _sanitize_numeric_value(self, value: Any) -> Any:
        """
        Rules:
        - scalar numeric or numeric-convertible -> keep as float/int-compatible value
        - scalar non-numeric -> None
        - list/tuple -> list where each non-numeric element becomes None
        """
        if isinstance(value, (list, tuple)):
            return [
                item if self._is_number_like(item) else None
                for item in value
            ]

        return value if self._is_number_like(value) else None

    @staticmethod
    def _is_number_like(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return False
        if isinstance(value, Number):
            return True
        try:
            float(value)
            return True

        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict:
        return {
            "score_calculation_time_nanos_history": self._sanitize_numeric_value(self.score_calculation_time_nanos_history),
            "removal_threshold_history": self._sanitize_numeric_value(self.removal_threshold_history),
            "aggregated_loss_history": self._sanitize_numeric_value(self.aggregated_loss_history),
            "average_accuracy_history": self._sanitize_numeric_value(self.average_accuracy_history),
            "average_accuracy_std_history": self._sanitize_numeric_value(self.average_accuracy_std_history),
            "average_r2_history": self._sanitize_numeric_value(self.average_r2_history),
            "average_loss_baseline_history": self._sanitize_numeric_value(self.average_loss_baseline_history),
            "average_mare_history": self._sanitize_numeric_value(self.average_mare_history),
            "average_mare_baseline_history": self._sanitize_numeric_value(self.average_mare_baseline_history),
            "tp_history": self._sanitize_numeric_value(self.tp_history),
            "tn_history": self._sanitize_numeric_value(self.tn_history),
            "fp_history": self._sanitize_numeric_value(self.fp_history),
            "fn_history": self._sanitize_numeric_value(self.fn_history),
            "total_fp_and_fn_history": self._sanitize_numeric_value(self.total_fp_and_fn_history),
            "removal_accuracy_history": self._sanitize_numeric_value(self.removal_accuracy_history),
            "removal_precision_history": self._sanitize_numeric_value(self.removal_precision_history),
            "removal_recall_history": self._sanitize_numeric_value(self.removal_recall_history),
            "removal_f1_history": self._sanitize_numeric_value(self.removal_f1_history),
            "mention_precision_history": self._sanitize_numeric_value(self.mention_precision_history),
            "mention_recall_history": self._sanitize_numeric_value(self.mention_recall_history),
            "mention_f1_history": self._sanitize_numeric_value(self.mention_f1_history),
            "document_precision_history": self._sanitize_numeric_value(self.document_precision_history),
            "document_recall_history": self._sanitize_numeric_value(self.document_recall_history),
            "document_f1_history": self._sanitize_numeric_value(self.document_f1_history),
        }

    @classmethod
    def from_dict(cls, data: dict, strategy_config: StrategyConfig) -> "RoundsInfo":
        obj = cls.__new__(cls)
        obj.simulation_strategy_config = strategy_config
        obj.__post_init__()  # sets up plottable_metrics etc.
        for key, value in data.items():
            setattr(obj, key, value)
        return obj