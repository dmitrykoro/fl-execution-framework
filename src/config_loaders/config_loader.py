import json
import logging
import os
import sys

from src.config_loaders.validate_strategy_config import validate_strategy_config
from src.utils.device_utils import calculate_optimal_gpus_per_client, get_device


class ConfigLoader:
    def __init__(self, usecase_config_path: str, dataset_config_path: str) -> None:
        self.usecase_config_path = os.path.join(usecase_config_path)
        self.usecase_config_list = self._merge_usecase_configs(self.usecase_config_path)

        self.dataset_config_path = os.path.join(dataset_config_path)
        self.dataset_config_list = self._set_config(self.dataset_config_path)

    def get_usecase_config_list(self) -> list:
        """Get config list"""

        return self.usecase_config_list

    def get_dataset_config_list(self) -> dict:
        """Get config of dataset folders based on dataset keywords"""

        return self.dataset_config_list

    def get_dataset_folder_name(self, key) -> str:
        """Get dataset folder name based on the JSON definition"""
        try:
            return self.dataset_config_list[key]
        except KeyError:
            logging.error(
                f"Error with the provided dataset key: {key}. Please specify it in {self.dataset_config_path}."
            )
            sys.exit(-1)

    @staticmethod
    def _merge_usecase_configs(config_path: str) -> list:
        """Merge usecase JSON to create full config for each strategy"""
        try:
            with open(config_path) as f:
                raw_config = json.load(f)
                logging.info(f"Successfully loaded confing for {config_path}.")

            shared_settings = raw_config["shared_settings"]

            for strategy in raw_config["simulation_strategies"]:
                strategy.update(shared_settings)

            for strategy in raw_config["simulation_strategies"]:
                # Validate the strategy configuration
                validate_strategy_config(strategy)
                logging.info(f"Successfully validated config from {config_path}.")

                # Convert training_device string to torch.device with CUDA fallback
                if "training_device" in strategy:
                    device_str = strategy["training_device"]
                    strategy["training_device"] = get_device(device_str)

                # Auto-calculate optimal GPU allocation if set to "auto" or -1
                if "gpus_per_client" in strategy:
                    gpu_value = strategy["gpus_per_client"]
                    # Support "auto" string and -1 as auto-detection triggers
                    if gpu_value == "auto" or gpu_value == -1 or gpu_value == "-1":
                        num_clients = strategy.get("num_of_clients", 5)
                        optimal_gpu = calculate_optimal_gpus_per_client(num_clients)
                        strategy["gpus_per_client"] = optimal_gpu

                        if optimal_gpu > 0:
                            max_parallel = int(1.0 / optimal_gpu)
                            logging.info(
                                f"[AUTO-ALLOCATE] Setting gpus_per_client={optimal_gpu:.2f} "
                                f"(allows {max_parallel} clients in parallel)"
                            )
                        else:
                            logging.info(
                                "[AUTO-ALLOCATE] No GPU available, setting gpus_per_client=0"
                            )

            return raw_config["simulation_strategies"]

        except Exception as e:
            logging.error(f"Error while loading config from JSON: {e}")
            sys.exit(-1)

    @staticmethod
    def _set_config(config_path: str) -> dict:
        """Set config by loading it from the specified JSON file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                logging.info(f"Successfully loaded confing for {config_path}.")

                return config

        except Exception as e:
            logging.error(f"Error while loading config from JSON: {e}")
            sys.exit(-1)
