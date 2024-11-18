import shutil
import os
import logging
import random


class DatasetHandler:

    def __init__(self, strategy_config, directory_handler, dataset_config_list) -> None:
        self._strategy_config = strategy_config

        self.dst_dataset = directory_handler.dataset_dir
        self.src_dataset = dataset_config_list[self._strategy_config.dataset_keyword]

    def setup_dataset(self) -> None:
        """Copy the specified number of clients' subsets to runtime folder and perform poisoning"""

        self._copy_dataset(self._strategy_config.num_of_clients)
        self._poison_clients(self._strategy_config.attack_type, self._strategy_config.num_of_malicious_clients)

    def teardown_dataset(self) -> None:
        """Remove dataset after execution if requested in the strategy config"""

        if not self._strategy_config.preserve_dataset:
            try:
                shutil.rmtree(self.dst_dataset)

            except Exception as e:
                logging.error(f"Error while cleaning up the dataset: {e}")

    def _copy_dataset(self, num_to_copy: str) -> None:
        """Copy dataset"""

        all_client_folders_list = [
            client_folder for client_folder in sorted(os.listdir(self.src_dataset))
            if (os.path.isdir(os.path.join(self.src_dataset, client_folder)) and not client_folder.startswith("."))
        ]
        client_folders_list = all_client_folders_list[:num_to_copy]

        for client_folder in client_folders_list:
            client_src = os.path.join(self.src_dataset, client_folder)
            client_dst = os.path.join(self.dst_dataset, client_folder)

            try:
                shutil.copytree(src=client_src, dst=client_dst)

            except Exception as e:
                logging.error(f"Error while preparing dataset: {e}")

    def _poison_clients(self, attack_type: str, num_to_poison: int) -> None:
        """Poison data according to the specified parameters in the strategy config"""

        client_dirs_to_poison = [
            client_dir for client_dir in sorted(os.listdir(self.dst_dataset)) if not client_dir.startswith(".")
        ][:num_to_poison]

        for client_dir in client_dirs_to_poison:
            if attack_type == "label_flipping":
                self._flip_labels(client_dir)
            else:
                raise NotImplementedError(f"Not supported attack type: {attack_type}")

    def _flip_labels(self, client_dir: str) -> None:
        """Perform 100% label flipping for the specified client"""

        available_labels = [
            label for label in os.listdir(os.path.join(self.dst_dataset, client_dir)) if not label.startswith(".")
        ]

        for label in available_labels:

            old_dir = os.path.join(self.dst_dataset, client_dir, label)
            new_dir = os.path.join(self.dst_dataset, client_dir, label + "_old")

            os.rename(old_dir, new_dir)

        for label in os.listdir(os.path.join(self.dst_dataset, client_dir)):
            if label.startswith('.'):  # skip .DS_store
                continue

            old_dir = os.path.join(self.dst_dataset, client_dir, label)

            while True:
                if len(available_labels) == 1:
                    new_label = available_labels[0]
                    break

                else:
                    new_label = random.choice(available_labels)

                    if new_label != label.split("_")[0]:
                        break

            new_dir = os.path.join(self.dst_dataset, client_dir, f'{new_label}')
            os.rename(old_dir, new_dir)
            available_labels.remove(new_label)

        os.rename(os.path.join(self.dst_dataset, client_dir), os.path.join(self.dst_dataset, client_dir + "_bad"))
