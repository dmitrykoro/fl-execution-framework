from dataclasses import dataclass, field
from typing import List, Union, Optional, Any
import numpy as np


@dataclass
class ClientInfo:

    client_id: int
    num_of_rounds: int
    is_malicious: bool = None
    rounds: List[int] = None

    _attack_schedule: List[dict] = None
    _legacy_attack_type: str = None
    _poisoned_client_ids: Optional[set[int]] = None

    attack_plan_keywords: List[List[str]] = None
    attack_plan_configs: List[dict] = field(default_factory=list)

    removal_criterion_history: List[Union[float, None]] = None
    absolute_distance_history: List[Union[float, None]] = None
    loss_history: List[Union[float, None]] = None
    accuracy_history: List[Union[float, None]] = None
    aggregation_participation_history: List[Union[int, None]] = None
    lqr_weight_history: List[Union[float, None]] = None

    plottable_metrics = [
        "removal_criterion_history",
        "absolute_distance_history",
        "loss_history",
        "accuracy_history",
    ]

    savable_metrics = [
        "removal_criterion_history",
        "absolute_distance_history",
        "loss_history",
        "accuracy_history",
        "aggregation_participation_history",
        "lqr_weight_history",
    ]

    def __post_init__(self):
        def _init_list(value=None):
            return [value] * self.num_of_rounds

        def _create_attack_plan(attack_schedule: list, legacy_attack_type: str):
            if legacy_attack_type:
                if self._poisoned_client_ids and self.client_id in self._poisoned_client_ids:
                    self.attack_plan_keywords = _init_list([legacy_attack_type])
                    self.is_malicious = True
                else:
                    self.attack_plan_keywords = _init_list([])
                return
            
            # If there is no attack type and not attack schedule (i.e no attack)
            elif not attack_schedule and not legacy_attack_type:
                self.attack_plan_keywords = _init_list([])
                return

            else:
                self.attack_plan_keywords = _init_list([])

            for attack_config in attack_schedule:
                malicious_client_ids = attack_config["malicious_client_ids"]

                if self.client_id not in malicious_client_ids:
                    continue

                self.is_malicious = True
                self.attack_plan_configs.append(attack_config)

                attack_type = attack_config["attack_type"]
                start_round = attack_config["start_round"]
                end_round = attack_config["end_round"]

                for idx, round_num in enumerate(range(1, self.num_of_rounds + 1)):
                    if start_round <= round_num <= end_round:
                        new_plan = list(self.attack_plan_keywords[idx])
                        new_plan.append(attack_type)
                        self.attack_plan_keywords[idx] = new_plan

        self.removal_criterion_history = _init_list()
        self.absolute_distance_history = _init_list()
        self.loss_history = _init_list()
        self.accuracy_history = _init_list()
        self.aggregation_participation_history = _init_list(value=1)
        self.lqr_weight_history = _init_list()

        _create_attack_plan(self._attack_schedule, self._legacy_attack_type)

        self.rounds = []

        for round_num in range(self.num_of_rounds):
            self.rounds.append(round_num + 1)

    def get_current_attack_keywords(self, current_round) -> list:
        """
        Get the attack scheduled for specified round.
        """
        return self.attack_plan_keywords[current_round - 1]

    def get_client_attack_configs(self) -> list:
        """
        Get configs for these attacks.
        """
        return self.attack_plan_configs

    def get_current_attack_configs(self, current_round) -> list:
        """
        Get the attack configs scheduled for specified round.
        """
        current_attack_configs = []
        for attack_config in self.attack_plan_configs:
            if (attack_config["start_round"] <= current_round <= attack_config["end_round"]):
                current_attack_configs.append(attack_config)
        return current_attack_configs
    
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
            current_round: int,
            removal_criterion: Optional[float] = None,
            absolute_distance: Optional[float] = None,
            loss: Optional[float] = None,
            accuracy: Optional[float] = None,
            aggregation_participation: Optional[int] = None,
            lqr_weight: Optional[float] = None,
    ) -> None:
        """
        Add client metrics history entry for a new round.

        :param current_round: specifies the round number to put all data into correct index at list
        :param accuracy: accuracy at this round
        :param loss: loss at this round
        :param removal_criterion: calculated score based on which the removal was or was not performed
        :param absolute_distance: to the centroid of all models
        :param aggregation_participation: 1 if client was aggregated, 0 if was not
        :param lqr_weight: QP-derived aggregation weight assigned by LQR-Fed (0-1)
        """

        if removal_criterion is not None:
            self.removal_criterion_history[current_round - 1] = self._to_python_number(removal_criterion)
        if absolute_distance is not None:
            self.absolute_distance_history[current_round - 1] = self._to_python_number(absolute_distance)
        if loss is not None:
            self.loss_history[current_round - 1] = self._to_python_number(loss)
        if accuracy is not None:
            self.accuracy_history[current_round - 1] = self._to_python_number(accuracy)
        if aggregation_participation is not None:
            self.aggregation_participation_history[current_round - 1] = aggregation_participation
        if lqr_weight is not None:
            self.lqr_weight_history[current_round - 1] = self._to_python_number(lqr_weight)

    def get_metric_by_name(self, metric: str) -> List:
        """Get single plottable or savable metric values by name"""

        return getattr(self, metric)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "num_of_rounds": self.num_of_rounds,
            "is_malicious": self.is_malicious,
            "attack_plan_keywords": self.attack_plan_keywords,
            "attack_plan_configs": self.attack_plan_configs,
            "removal_criterion_history": self.removal_criterion_history,
            "absolute_distance_history": self.absolute_distance_history,
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "aggregation_participation_history": self.aggregation_participation_history,
            "lqr_weight_history": self.lqr_weight_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClientInfo":
        obj = cls.__new__(cls)
        for key, value in data.items():
            setattr(obj, key, value)
        obj.rounds = list(range(1, obj.num_of_rounds + 1))
        return obj