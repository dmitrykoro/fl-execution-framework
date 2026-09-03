import os
import platform

import numpy as np
import psutil
import torch
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data_models.client_info import ClientInfo
from src.data_models.round_info import RoundsInfo
from src.data_models.simulation_strategy_config import StrategyConfig
from src.dataset_handlers.dataset_handler import DatasetHandler


@dataclass
class SimulationStrategyHistory:

    strategy_config: StrategyConfig
    dataset_handler: DatasetHandler
    rounds_history: RoundsInfo = field(init=False)
    total_malicious_clients: int = 0
    malicious_clients_ids: set = field(default_factory=set)
    _clients_dict: dict = field(default_factory=dict)
    training_device_type: str = "cpu"
    training_device_name: str = "cpu"
    training_device_ram: str = None
    training_device_arch: str = None
    host_cpu: str = None
    host_ram: str = None

    def __post_init__(self):
        self.rounds_history = RoundsInfo(
            simulation_strategy_config=self.strategy_config
        )

        for i in range(self.strategy_config.num_of_clients):
            self._clients_dict[i] = ClientInfo(
                client_id=i,
                num_of_rounds=self.strategy_config.num_of_rounds,
                is_malicious=False,
                _attack_schedule=(
                    self.strategy_config.attack_schedule
                    if self.strategy_config.attack_schedule
                    else None
                ),
                _legacy_attack_type=(
                    self.strategy_config.attack_type
                    if self.strategy_config.attack_type
                    else None
                ),
                _poisoned_client_ids=self.dataset_handler.poisoned_client_ids,
                model_type=self.strategy_config.model_type,
            )

        for client in self._clients_dict.values():
            if client.is_malicious:
                self.total_malicious_clients += 1
                self.malicious_clients_ids.add(client.client_id)

        self.host_cpu = platform.processor() or platform.machine()
        self.host_ram = str(psutil.virtual_memory().total)
        self.training_device_arch = platform.system()

        device = self.strategy_config.training_device
        device_type = str(device).split(":")[0]
        if device_type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            self.training_device_type = "gpu"
            self.training_device_name = props.name
            self.training_device_ram = f"{props.total_memory / (1024 ** 3):.1f} GB"
            self.training_device_arch = f"sm_{props.major}{props.minor}"
        else:
            self.training_device_type = "cpu"
            self.training_device_name = "cpu"
            self.training_device_ram = None

    def get_all_clients(self) -> list:
        """Return a list of all ClientInfo instances."""
        return list(self._clients_dict.values())

    def get_client_by_id(self, client_id: int) -> ClientInfo:
        """Return a ClientInfo instance by its ID."""
        return self._clients_dict.get(client_id)

    def get_malicious_client_ids(self) -> set[int]:
        return self.malicious_clients_ids

    @staticmethod
    def _to_python_number(value: Any) -> Any:
        """
        Convert value to python number to allow for later database upload.
        Catches np.float32, np.float64, etc. which are not json serializable.
        """
        if value is not None and isinstance(value, np.generic):
            return value.item()
        return value

    def insert_single_client_history_entry(
        self,
        client_id: int,
        current_round: int,
        removal_criterion: float = None,
        absolute_distance: float = None,
        loss: float = None,
        accuracy: float = None,
        ss_res: float = None,
        ss_tot: float = None,
        r2: float = None,
        loss_baseline: float = None,
        mare: float = None,
        mare_baseline: float = None,
        num_elements: int = None,
        aggregation_participation: int = None,
        lqr_weight: float = None,
    ) -> None:
        """Insert a history entry for a single client.

        Only the values explicitly provided will be updated.
        """
        updating_client: ClientInfo = self._clients_dict[client_id]
        updating_client.add_history_entry(
            current_round,
            self._to_python_number(removal_criterion),
            self._to_python_number(absolute_distance),
            self._to_python_number(loss),
            self._to_python_number(accuracy),
            self._to_python_number(ss_res),
            self._to_python_number(ss_tot),
            self._to_python_number(r2),
            self._to_python_number(loss_baseline),
            self._to_python_number(mare),
            self._to_python_number(mare_baseline),
            self._to_python_number(num_elements),
            aggregation_participation,
            self._to_python_number(lqr_weight),
        )

    def insert_round_history_entry(
        self,
        score_calculation_time_nanos: int = None,
        removal_threshold: float = None,
        loss_aggregated: float = None,
    ) -> None:
        """Append round history info. Only the values explicitly provided will be updated."""
        if score_calculation_time_nanos is not None:
            self.rounds_history.score_calculation_time_nanos_history.append(
                score_calculation_time_nanos
            )
        if removal_threshold is not None:
            self.rounds_history.removal_threshold_history.append(self._to_python_number(removal_threshold))
        if loss_aggregated is not None:
            self.rounds_history.aggregated_loss_history.append(self._to_python_number(loss_aggregated))

    def update_client_participation(
        self,
        current_round: int,
        removed_client_ids: set,
    ) -> None:
        """Update client participation history based on removed client IDs for a given round."""
        for client_id in removed_client_ids:
            self.insert_single_client_history_entry(
                client_id=int(client_id),
                current_round=current_round,
                aggregation_participation=0,
            )

    def calculate_additional_rounds_data(self) -> None:
        """Calculate derivative per-round metrics from primary simulation data.

        Primary data collected during simulation:

        Per-client data over rounds (stored in ClientInfo):
            - loss_history
            - accuracy_history: achieved accuracy during validation
            - removal_criterion_history: value used to determine removal
            - absolute_distance_history: distance to the cluster center

        Per-round data over rounds (stored in RoundsInfo):
            - aggregated_loss_history: loss of all aggregated client models
            - score_calculation_time_nanos_history: time to calculate removal_criterion
            - removal_threshold_history: exclusion threshold at each round

        Derivative data calculated here (per round):
            - average_accuracy_history: average accuracy of benign (non-attacking) clients
            - average_r2_history: average R2 of benign (non-attacking) clients (only for regression task datasets)
        """
        for round_num in range(self.strategy_config.num_of_rounds):
            round_tp_count = 0
            round_tn_count = 0
            round_fp_count = 0
            round_fn_count = 0

            round_ss_res = 0.0
            round_ss_tot = 0.0

            round_loss_baseline = 0.0
            round_mare = 0.0
            round_mare_baseline = 0.0
            round_num_elements = 0

            num_aggregated_clients = 0
            sum_aggregated_accuracies = 0
            round_client_accuracies = []

            for client_info in self.get_all_clients():
                # round_num is a 0-based loop index; get_current_attack_keywords
                # takes a 1-based round, as every other call site passes.
                client_is_malicious = client_info.get_current_attack_keywords(round_num + 1)
                client_was_aggregated = (
                    client_info.aggregation_participation_history[round_num] == 1
                )

                if self.strategy_config.remove_clients:
                    if not client_is_malicious and client_was_aggregated:
                        round_tp_count += 1
                    if client_is_malicious and not client_was_aggregated:
                        round_tn_count += 1
                    if not client_is_malicious and not client_was_aggregated:
                        round_fp_count += 1
                    if client_is_malicious and client_was_aggregated:
                        round_fn_count += 1

                client_is_attacking_this_round = bool(client_is_malicious)
                if not client_is_attacking_this_round:
                    accuracy = client_info.accuracy_history[round_num]
                    if accuracy is not None:
                        num_aggregated_clients += 1
                        sum_aggregated_accuracies += accuracy
                        round_client_accuracies.append(accuracy)

                    ss_res = client_info.ss_res_history[round_num]
                    ss_tot = client_info.ss_tot_history[round_num]
                    if ss_res is not None and ss_tot is not None:
                        round_ss_res += ss_res
                        round_ss_tot += ss_tot

                    loss_baseline = client_info.loss_baseline_history[round_num]
                    mare = client_info.mare_history[round_num]
                    mare_baseline = client_info.mare_baseline_history[round_num]
                    num_elements = client_info.num_elements_history[round_num]
                    if (loss_baseline is not None and mare is not None 
                        and mare_baseline is not None and num_elements is not None):
                        round_loss_baseline += loss_baseline * num_elements
                        round_mare += mare * num_elements
                        round_mare_baseline += mare_baseline * num_elements
                        round_num_elements += num_elements

            self.rounds_history.append_tp_tn_fp_fn(
                round_tp_count, round_tn_count, round_fp_count, round_fn_count
            )
            self.rounds_history.average_accuracy_history.append(
                float(
                    f"{(sum_aggregated_accuracies / num_aggregated_clients * 100):.2f}"
                )
                if num_aggregated_clients > 0
                else 0.000
            )
            self.rounds_history.average_accuracy_std_history.append(
                float(f"{np.std(round_client_accuracies) * 100:.2f}")
                if len(round_client_accuracies) > 1
                else 0.00
            )

            # calculate aggregate R^2 for the round
            global_r2 = 1 - (round_ss_res / round_ss_tot) if round_ss_tot > 0 else 0.0
            self.rounds_history.average_r2_history.append(self._to_python_number(global_r2))

            # calculate aggregate baseline loss, MARE, and baseline MARE through weighted average
            global_baseline_loss = round_loss_baseline / round_num_elements if round_num_elements > 0 else 0.0
            self.rounds_history.average_loss_baseline_history.append(self._to_python_number(global_baseline_loss))
            global_mare = round_mare / round_num_elements if round_num_elements > 0 else 0.0
            self.rounds_history.average_mare_history.append(self._to_python_number(global_mare))
            global_mare_baseline = round_mare_baseline / round_num_elements if round_num_elements > 0 else 0.0
            self.rounds_history.average_mare_baseline_history.append(self._to_python_number(global_mare_baseline))

        if self.strategy_config.remove_clients:
            self.rounds_history.calculate_additional_metrics()

    def to_dict(self) -> dict:
        return {
            "strategy_config": vars(self.strategy_config),
            "total_malicious_clients": self.total_malicious_clients,
            "malicious_clients_ids": list(self.malicious_clients_ids),
            "rounds_history": self.rounds_history.to_dict(),
            "clients": [c.to_dict() for c in self._clients_dict.values()],
            "training_device_type": self.training_device_type,
            "training_device_name": self.training_device_name,
            "training_device_ram": self.training_device_ram,
            "training_device_arch": self.training_device_arch,
            "host_cpu": self.host_cpu,
            "host_ram": self.host_ram,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationStrategyHistory":
        strategy_config = StrategyConfig.from_dict(data["strategy_config"])
        obj = cls.__new__(cls)
        obj.strategy_config = strategy_config
        obj.total_malicious_clients = data["total_malicious_clients"]
        obj.malicious_clients_ids = set(data["malicious_clients_ids"])
        obj.rounds_history = RoundsInfo.from_dict(data["rounds_history"], strategy_config)
        obj._clients_dict = {
            c["client_id"]: ClientInfo.from_dict(c, model_type=strategy_config.model_type)
            for c in data["clients"]
        }
        obj.training_device_type = data.get("training_device_type", "cpu")
        obj.training_device_name = data.get("training_device_name", "cpu")
        obj.training_device_ram = data.get("training_device_ram", None)
        obj.training_device_arch = data.get("training_device_arch", None)
        obj.host_cpu = data.get("host_cpu", None)
        obj.host_ram = data.get("host_ram", None)
        return obj