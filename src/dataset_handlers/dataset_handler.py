import shutil
import os
import logging


class DatasetHandler:

    def __init__(self, strategy_config, directory_handler, dataset_config_list) -> None:
        self._strategy_config = strategy_config

        self.dst_dataset = directory_handler.dataset_dir
        self.src_dataset = dataset_config_list[self._strategy_config.dataset_keyword]

    def setup_dataset(self) -> None:
        """Copy the specified number of clients' subsets to runtime folder"""
        self._copy_dataset(self._strategy_config.num_of_clients)

    def teardown_dataset(self) -> None:
        """Remove dataset after execution if requested in the strategy config"""

        if not self._strategy_config.preserve_dataset:
            try:
                shutil.rmtree(self.dst_dataset)

            except Exception as e:
                logging.error(f"Error while cleaning up the dataset: {e}")

    def _copy_dataset(self, num_to_copy: int) -> None:
        """Copy dataset"""

        all_client_folders_list = [
            client_folder for client_folder in sorted(os.listdir(self.src_dataset))
            if (os.path.isdir(os.path.join(self.src_dataset, client_folder)) and not client_folder.startswith("."))
        ]
        all_client_folders_list = sorted(all_client_folders_list, key=lambda string: int(string.split("_")[1]))

        client_folders_list = all_client_folders_list[:num_to_copy]

        for client_folder in client_folders_list:
            client_src = os.path.join(self.src_dataset, client_folder)
            client_dst = os.path.join(self.dst_dataset, client_folder)

            try:
                shutil.copytree(src=client_src, dst=client_dst)

            except Exception as e:
                logging.error(f"Error while preparing dataset: {e}")
