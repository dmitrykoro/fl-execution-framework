import os
import json
import logging
import sys

from src.config_loaders.validate_strategy_config import validate_strategy_config


class ConfigLoader:
    def __init__(
            self,
            usecase_config_path: str,
            dataset_config_path: str
    ) -> None:
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
            logging.error(f"Error with the provided dataset key: {key}. Please specify it in {self.dataset_config_path}.")
            sys.exit(-1)

    @staticmethod
    def _merge_usecase_configs(config_path: str) -> list:
        """Merge usecase JSON to create full config for each strategy"""
        try:
            with open(config_path) as f:
                raw_config = json.load(f)
                logging.info(f"Successfully loaded config for {config_path}.")

            shared_settings = raw_config['shared_settings']

            for strategy in raw_config['simulation_strategies']:
                strategy.update(shared_settings)

            for strategy in raw_config['simulation_strategies']:
                # Validate the strategy configuration
                validate_strategy_config(strategy)
                logging.info(f"Successfully validated config from {config_path}.")

            # Log attack schedule info
            if raw_config['simulation_strategies'] and raw_config['simulation_strategies'][0].get('attack_schedule'):
                attack_schedule = raw_config['simulation_strategies'][0]['attack_schedule']
                for idx, entry in enumerate(attack_schedule):
                    if '_selected_clients' in entry:
                        logging.info(
                            f"attack_schedule entry {idx} ({entry.get('selection_strategy')}): "
                            f"Selected clients {entry['_selected_clients']} for {entry.get('attack_type')} attack"
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
                logging.info(f"Successfully loaded config for {config_path}.")

                return config

        except Exception as e:
            logging.error(f"Error while loading config from JSON: {e}")
            sys.exit(-1)
