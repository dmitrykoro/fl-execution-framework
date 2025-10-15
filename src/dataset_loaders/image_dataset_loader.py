import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class ImageDatasetLoader:
    def __init__(
        self,
        transformer: transforms,
        dataset_dir: str,
        num_of_clients: int,
        batch_size: int,
        training_subset_fraction: float,
    ) -> None:
        self.transformer = transformer
        self.dataset_dir = dataset_dir
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction

    def load_datasets(self):
        """Partition and load the dataset for each client."""
        trainloaders = []
        valloaders = []

        client_folders = [
            d for d in os.listdir(self.dataset_dir) if d.startswith("client_")
        ]

        for client_folder in sorted(
            client_folders, key=lambda string: int(string.split("_")[1])
        ):
            client_dataset = datasets.ImageFolder(
                root=self.dataset_dir + f"/{client_folder}", transform=self.transformer
            )

            train_subset_len = int(len(client_dataset) * self.training_subset_fraction)
            val_subset_len = len(client_dataset) - train_subset_len

            # Ensure at least 1 sample in validation if possible
            if val_subset_len == 0 and len(client_dataset) > 1:
                train_subset_len -= 1
                val_subset_len = 1

            train_dataset, validation_dataset = random_split(
                client_dataset,
                [train_subset_len, val_subset_len],
                torch.Generator().manual_seed(42),
            )

            trainloaders.append(
                DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            )
            valloaders.append(
                DataLoader(validation_dataset, batch_size=self.batch_size)
            )

        return trainloaders, valloaders
