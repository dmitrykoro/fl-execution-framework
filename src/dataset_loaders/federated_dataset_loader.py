from typing import List, Optional, Tuple

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader, random_split


class FederatedDatasetLoader:
    """
    Load datasets from HuggingFace Hub using flwr-datasets library.

    Supports IID, Dirichlet, and Pathological partitioning strategies.
    """

    def __init__(
        self,
        dataset_name: str,
        num_of_clients: int,
        batch_size: int,
        training_subset_fraction: float,
        partitioning_strategy: str = "iid",
        partitioning_params: Optional[dict] = None,
        label_column: str = "label",
    ) -> None:
        """
        Initialize FederatedDatasetLoader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "mnist", "cifar10")
            num_of_clients: Number of federated learning clients
            batch_size: Batch size for DataLoaders
            training_subset_fraction: Fraction of data for training (0.0-1.0)
            partitioning_strategy: "iid", "dirichlet", or "pathological"
            partitioning_params: Strategy-specific parameters (e.g., {"alpha": 0.5})
            label_column: Name of the label column in the dataset (default: "label")
        """
        self.dataset_name = dataset_name
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.training_subset_fraction = training_subset_fraction
        self.partitioning_strategy = partitioning_strategy
        self.partitioning_params = partitioning_params or {}
        self.label_column = label_column

    def load_datasets(self) -> Tuple[List[DataLoader], List[DataLoader], Optional[int]]:
        """
        Load and partition dataset from HuggingFace Hub.

        Returns:
            tuple: (trainloaders, valloaders, num_classes)
                - trainloaders: List of PyTorch DataLoaders for training
                - valloaders: List of PyTorch DataLoaders for validation
                - num_classes: Number of classification labels (None if not detected)
        """
        # Create partitioner based on strategy
        partitioner = self._create_partitioner()

        # Load federated dataset from HuggingFace Hub
        fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

        trainloaders = []
        valloaders = []
        num_classes = None

        # Try to detect number of classes from the dataset features
        try:
            if (
                hasattr(fds.dataset, "features")
                and self.label_column in fds.dataset.features
            ):
                label_feature = fds.dataset.features[self.label_column]
                if hasattr(label_feature, "names"):
                    num_classes = len(label_feature.names)
        except Exception:
            # If detection fails, num_classes remains None
            pass

        for client_id in range(self.num_of_clients):
            # Load partition for this client
            partition = fds.load_partition(partition_id=client_id, split="train")
            partition.set_format("torch")

            # Split into train/val
            train_size = int(len(partition) * self.training_subset_fraction)
            val_size = len(partition) - train_size

            # Ensure at least 1 sample in validation if possible
            if val_size == 0 and len(partition) > 1:
                train_size -= 1
                val_size = 1

            train_dataset, val_dataset = random_split(
                partition, [train_size, val_size], torch.Generator().manual_seed(42)
            )

            trainloaders.append(
                DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            )
            valloaders.append(DataLoader(val_dataset, batch_size=self.batch_size))

        return trainloaders, valloaders, num_classes

    def _create_partitioner(self):
        """
        Create partitioner based on strategy.

        Returns:
            Partitioner: flwr-datasets partitioner instance

        Raises:
            ValueError: If partitioning_strategy is unknown
        """
        if self.partitioning_strategy == "iid":
            return IidPartitioner(num_partitions=self.num_of_clients)

        elif self.partitioning_strategy == "dirichlet":
            alpha = self.partitioning_params.get("alpha", 0.5)
            return DirichletPartitioner(
                num_partitions=self.num_of_clients,
                partition_by=self.label_column,
                alpha=alpha,
            )

        elif self.partitioning_strategy == "pathological":
            num_classes = self.partitioning_params.get("num_classes_per_partition", 2)
            return PathologicalPartitioner(
                num_partitions=self.num_of_clients,
                partition_by=self.label_column,
                num_classes_per_partition=num_classes,
            )

        else:
            raise ValueError(
                f"Unknown partitioning strategy: {self.partitioning_strategy}"
            )
