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

        num_classes = self._detect_num_classes(fds)

        trainloaders = []
        valloaders = []

        for client_id in range(self.num_of_clients):
            partition = fds.load_partition(partition_id=client_id, split="train")

            partition = self._standardize_columns(partition)
            partition.set_format("torch")

            train_size = int(len(partition) * self.training_subset_fraction)
            val_size = len(partition) - train_size

            # Ensure at least 1 sample in validation if possible
            if val_size == 0 and len(partition) > 1:
                train_size -= 1
                val_size = 1

            train_dataset, val_dataset = random_split(
                partition, [train_size, val_size], torch.Generator().manual_seed(42)
            )

            use_cuda = torch.cuda.is_available()
            trainloaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=use_cuda,
                    persistent_workers=True,
                )
            )
            valloaders.append(
                DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    num_workers=2,
                    pin_memory=use_cuda,
                    persistent_workers=True,
                )
            )

        return trainloaders, valloaders, num_classes

    def _detect_num_classes(self, fds: FederatedDataset) -> Optional[int]:
        """
        Detect number of classes from dataset features.

        Args:
            fds: FederatedDataset instance

        Returns:
            int: Number of classes, or None if not detected
        """
        try:
            if (
                hasattr(fds.dataset, "features")
                and self.label_column in fds.dataset.features
            ):
                label_feature = fds.dataset.features[self.label_column]
                if hasattr(label_feature, "names"):
                    return len(label_feature.names)
        except Exception:
            pass
        return None

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

    def _standardize_columns(self, dataset):
        """
        Rename dataset columns to standard format and remove extra columns.

        Standard format matches transformers library conventions:
            - Images: "pixel_values"
            - Labels: "labels" (plural)
            - Text: "input_ids", "attention_mask", "labels"

        Args:
            dataset: HuggingFace dataset partition

        Returns:
            dataset: Dataset with standardized column names and only required columns
        """
        rename_mapping = {}

        # Detect and rename image column
        image_cols = ["image", "img"]
        for col in image_cols:
            if col in dataset.column_names and col != "pixel_values":
                rename_mapping[col] = "pixel_values"
                break

        # Detect and rename label column
        # Priority: self.label_column > common label columns
        label_col = None
        if self.label_column in dataset.column_names:
            label_col = self.label_column
        else:
            # Fallback: search for common label columns
            for col in ["label", "character", "fine_label"]:
                if col in dataset.column_names:
                    label_col = col
                    break

        if label_col and label_col != "labels":
            rename_mapping[label_col] = "labels"

        # Apply renaming if needed
        if rename_mapping:
            dataset = dataset.rename_columns(rename_mapping)

        # Remove extra columns - keep only standard ones
        # Image datasets: pixel_values + labels
        # Text datasets: input_ids + attention_mask + labels
        standard_columns = {"pixel_values", "labels", "input_ids", "attention_mask"}
        columns_to_remove = [
            col for col in dataset.column_names if col not in standard_columns
        ]

        if columns_to_remove:
            dataset = dataset.remove_columns(columns_to_remove)

        return dataset
