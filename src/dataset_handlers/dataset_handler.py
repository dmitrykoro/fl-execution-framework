import logging
import os
import random
import shutil
import sys

import cv2
import numpy as np


class DatasetHandler:
    def __init__(self, strategy_config, directory_handler, dataset_config_list) -> None:
        self._strategy_config = strategy_config

        self.dst_dataset = directory_handler.dataset_dir

        # For HuggingFace datasets, dataset_keyword is None - skip local dataset lookup
        if self._strategy_config.dataset_keyword is not None:
            self.src_dataset = dataset_config_list[
                self._strategy_config.dataset_keyword
            ]
        else:
            self.src_dataset = None

        self.poisoned_client_ids = set()
        self.all_poisoned_img_snrs = []

        # Identify malicious clients for HuggingFace datasets
        self._identify_malicious_clients_for_hf_datasets()

    def _identify_malicious_clients_for_hf_datasets(self) -> None:
        """Identify malicious clients for HuggingFace datasets based on config."""
        if self._strategy_config.dataset_source != "huggingface":
            return

        # Static attacks: first N clients are malicious (client IDs 0 to N-1)
        if (
            not self._strategy_config.dynamic_attacks
            or not self._strategy_config.dynamic_attacks.get("enabled")
        ):
            for i in range(self._strategy_config.num_of_malicious_clients):
                self.poisoned_client_ids.add(i)
        else:
            # Dynamic attacks: extract client IDs from attack schedule
            schedule = self._strategy_config.dynamic_attacks.get("schedule", [])
            for phase in schedule:
                if phase.get("selection_strategy") == "specific":
                    client_ids = phase.get("client_ids", [])
                    self.poisoned_client_ids.update(client_ids)

    def setup_dataset(self) -> None:
        """Copy the specified number of clients' subsets to runtime folder and perform poisoning"""

        if self.src_dataset is None:
            logging.info("Skipping dataset setup for HuggingFace dataset")
            return

        self._copy_dataset(self._strategy_config.num_of_clients)
        self._poison_clients(
            self._strategy_config.attack_type,
            self._strategy_config.num_of_malicious_clients,
        )

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
            client_folder
            for client_folder in sorted(os.listdir(self.src_dataset))
            if (
                os.path.isdir(os.path.join(self.src_dataset, client_folder))
                and not client_folder.startswith(".")
            )
        ]
        all_client_folders_list = sorted(
            all_client_folders_list, key=lambda string: int(string.split("_")[1])
        )

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
            client_dir
            for client_dir in sorted(
                os.listdir(self.dst_dataset),
                key=lambda string: int(string.split("_")[1]),
            )
            if not client_dir.startswith(".")
        ][:num_to_poison]

        self._assign_poisoned_client_ids(client_dirs_to_poison)

        for client_dir in client_dirs_to_poison:
            if attack_type == "label_flipping":
                self._flip_labels(client_dir)
            elif attack_type == "gaussian_noise":
                self._add_noise(client_dir)
            else:
                raise NotImplementedError(f"Not supported attack type: {attack_type}")

        if attack_type == "gaussian_noise" and num_to_poison > 0:
            if len(self.all_poisoned_img_snrs) > 0:
                logging.warning(
                    f"Avg. SNR for poisoned images: {np.average(self.all_poisoned_img_snrs)}"
                )

    def _flip_labels(self, client_dir: str) -> None:
        """Perform 100% label flipping for the specified client"""

        available_labels = [
            label
            for label in os.listdir(os.path.join(self.dst_dataset, client_dir))
            if not label.startswith(".")
        ]

        for label in available_labels:
            old_dir = os.path.join(self.dst_dataset, client_dir, label)
            new_dir = os.path.join(self.dst_dataset, client_dir, label + "_old")

            os.rename(old_dir, new_dir)

        for label in os.listdir(os.path.join(self.dst_dataset, client_dir)):
            if label.startswith("."):  # skip .DS_store
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

            new_dir = os.path.join(self.dst_dataset, client_dir, f"{new_label}")
            os.rename(old_dir, new_dir)
            available_labels.remove(new_label)

        os.rename(
            os.path.join(self.dst_dataset, client_dir),
            os.path.join(self.dst_dataset, client_dir + "_bad"),
        )

    def _add_noise(self, client_dir: str) -> None:
        """Add Gaussian noise to a subset of images for the specified client."""

        supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        client_path = os.path.join(self.dst_dataset, client_dir)

        try:
            label_folders = [
                d
                for d in os.listdir(client_path)
                if os.path.isdir(os.path.join(client_path, d)) and not d.startswith(".")
            ]
        except FileNotFoundError:
            logging.error(f"Client directory not found: {client_path}")
            return

        for label_folder in label_folders:
            label_path = os.path.join(client_path, label_folder)

            all_images = [
                f
                for f in os.listdir(label_path)
                if f.lower().endswith(supported_extensions)
            ]

            if not all_images:
                logging.warning(f"No valid images in {label_path}")
                continue

            split_index = int(len(all_images) * self._strategy_config.attack_ratio)
            images_to_corrupt = all_images[:split_index]

            for filename in images_to_corrupt:
                filepath = os.path.join(label_path, filename)
                image = cv2.imread(filepath)

                if image is None:
                    logging.error(f"Failed to load image: {filepath}")
                    continue

                mean = self._strategy_config.gaussian_noise_mean
                std = self._strategy_config.gaussian_noise_std
                noise = np.random.normal(mean, std, image.shape).astype(np.float32)

                signal_power = np.mean(image.astype(np.float32) ** 2)
                noise_power = np.mean(noise**2)

                """
                Signal-to-Noise Ratio (SNR) in decibels (dB) is calculated as:
                
                    SNR (dB) = 10 * log10(signal_power / noise_power)
                
                where:
                    signal_power = mean squared value of the original image
                    noise_power  = mean squared value of the added noise
                """

                snr = 10 * np.log10(signal_power / noise_power)
                self.all_poisoned_img_snrs.append(snr)

                noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(
                    np.uint8
                )
                success = cv2.imwrite(filepath, noisy_image)

                if success:
                    logging.info(f"Added noise to: {filepath}")
                else:
                    logging.error(f"Failed to write image: {filepath}")

    def _assign_poisoned_client_ids(self, bad_client_dirs: list) -> None:
        """Assign ids of poisoned clients to class field"""

        for bad_client_dir in bad_client_dirs:
            try:
                self.poisoned_client_ids.add(int(bad_client_dir.split("_")[1]))
            except Exception as e:
                logging.error(
                    f"Error while parsing client dataset folder: client id must be a number: {e}"
                )
                sys.exit(-1)
