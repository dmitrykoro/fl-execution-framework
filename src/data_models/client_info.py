from dataclasses import dataclass, field
from typing import List, Union, Optional


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

        def _create_attack_plan(attack_schedule: list, legacy_attack_type: str):
            if legacy_attack_type:
                if self._poisoned_client_ids and self.client_id in self._poisoned_client_ids:
                    self.attack_plan_keywords = _init_list([legacy_attack_type])
                    self.is_malicious = True
                else:
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
        self.ss_res_history = _init_list()
        self.ss_tot_history = _init_list()
        self.r2_history = _init_list()
        self.aggregation_participation_history = _init_list(value=1)

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
        Get the attack configs scheduled for specified round based on current round attack keywords.
        """
        current_keywords = self.get_current_attack_keywords(current_round)
        current_attack_configs = []
        for attack_keyword in current_keywords:
            for attack_config in self.attack_plan_configs:
                if (attack_keyword == attack_config["attack_type"]
                    and attack_config["start_round"] <= current_round <= attack_config["end_round"]):
                    current_attack_configs.append(attack_config)
        return current_attack_configs
        

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
            "ss_res_history": self.ss_res_history,
            "ss_tot_history": self.ss_tot_history,
            "r2_history": self.r2_history,
            "aggregation_participation_history": self.aggregation_participation_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClientInfo":
        obj = cls.__new__(cls)
        for key, value in data.items():
            setattr(obj, key, value)
        obj.rounds = list(range(1, obj.num_of_rounds + 1))
        return obj