import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


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
        """Function to partition and load the dataset for each client."""

        trainloaders = []
        valloaders = []

        for i in range(self.num_of_clients):
            client_folder = os.path.join(self.dataset_dir, f'client_{i}')
            client_dataset = datasets.ImageFolder(root=client_folder, transform=self.transformer)

            train_subset_len = int(len(client_dataset) * self.training_subset_fraction)
            val_subset_len = int(len(client_dataset) - train_subset_len)

            train_dataset, validation_dataset = random_split(
                client_dataset,
                [train_subset_len, val_subset_len],
                torch.Generator().manual_seed(42)
            )

            trainloaders.append(DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True))
            valloaders.append(DataLoader(validation_dataset, batch_size=self.batch_size))

        return trainloaders, valloaders
