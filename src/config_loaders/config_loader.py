import os
import json
import logging


class ConfigLoader:
    def __init__(
            self,
            usecase_config_path: str,
            dataset_config_path: str
    ) -> None:
        self.usecase_config_path = os.path.join(usecase_config_path)
        self.usecase_config_list = self._merge_usecase_configs(self.usecase_config_path)

        self.dataset_config_path = os.path.join(dataset_config_path)
        self.dataset_config_dict = self._set_config(self.dataset_config_path)

    def get_usecase_config_list(self) -> list:
        """Get config list"""
        return self.usecase_config_list

    def get_dataset_folder_name(self, key) -> str:
        """Get dataset folder name based on the JSON definition"""
        try:
            return self.dataset_config_dict[key]
        except KeyError:
            print(f"Error with the provided dataset key: {key}. Please specify it in {self.dataset_config_path}.")

    @staticmethod
    def _merge_usecase_configs(config_path: str) -> list:
        """Merge usecase JSON to create full config for each strategy"""
        try:
            with open(config_path) as f:
                raw_config = json.load(f)
                logging.info(f"Successfully loaded confing for {config_path}.")

            shared_settings = raw_config['shared_settings']

            for strategy in raw_config['simulation_strategies']:
                strategy.update(shared_settings)

            return raw_config["simulation_strategies"]

        except Exception as e:
            logging.error(f"Error while loading config from JSON: {e}")

    @staticmethod
    def _set_config(config_path: str) -> list:
        """Set config by loading it from the specified JSON file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                logging.info(f"Successfully loaded confing for {config_path}.")

                return config

        except Exception as e:
            logging.error(f"Error while loading config from JSON: {e}")
