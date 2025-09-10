"""
Mock dataset generators and utilities for testing federated learning components.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MockDataset(Dataset):
    """Lightweight mock dataset for testing without real data dependencies."""

    def __init__(
        self,
        size: int = 100,
        num_classes: int = 10,
        input_shape: Tuple[int, ...] = (3, 32, 32),
    ):
        """
        Initialize mock dataset with specified parameters.

        Args:
            size: Number of samples in the dataset
            num_classes: Number of classification classes
            input_shape: Shape of input data (channels, height, width)
        """
        self.size = size
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Generate reproducible mock data
        torch.manual_seed(42)
        np.random.seed(42)

        # Create mock data and labels
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class MockFederatedDataset:
    """Mock federated dataset that simulates client data distribution."""

    def __init__(
        self,
        num_clients: int = 10,
        samples_per_client: int = 50,
        num_classes: int = 10,
        input_shape: Tuple[int, ...] = (3, 32, 32),
    ):
        """
        Initialize mock federated dataset.

        Args:
            num_clients: Number of federated clients
            samples_per_client: Number of samples per client
            num_classes: Number of classification classes
            input_shape: Shape of input data
        """
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Generate client datasets
        self.client_datasets = self._generate_client_datasets()

    def _generate_client_datasets(self) -> Dict[int, MockDataset]:
        """Generate individual datasets for each client."""
        client_datasets = {}

        for client_id in range(self.num_clients):
            # Use different seeds for each client to simulate data heterogeneity
            torch.manual_seed(42 + client_id)
            np.random.seed(42 + client_id)

            client_datasets[client_id] = MockDataset(
                size=self.samples_per_client,
                num_classes=self.num_classes,
                input_shape=self.input_shape,
            )

        return client_datasets

    def get_client_dataset(self, client_id: int) -> MockDataset:
        """Get dataset for a specific client."""
        if client_id not in self.client_datasets:
            raise ValueError(f"Client {client_id} not found")
        return self.client_datasets[client_id]

    def get_client_dataloader(self, client_id: int, batch_size: int = 32) -> DataLoader:
        """Get DataLoader for a specific client."""
        dataset = self.get_client_dataset(client_id)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class MockDatasetHandler:
    """Mock dataset handler that simulates dataset setup and teardown operations."""

    def __init__(self, dataset_type: str = "mock", dataset_path: str = "/tmp/mock"):
        """
        Initialize mock dataset handler.

        Args:
            dataset_type: Type of dataset (for compatibility)
            dataset_path: Path to dataset (mocked)
        """
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.is_setup = False
        self.federated_dataset: Optional[MockFederatedDataset] = None
        self.poisoned_client_ids = set()

    def setup_dataset(self, num_clients: int = 10) -> None:
        """Mock dataset setup without file operations."""
        # Simulate dataset setup based on type
        input_shapes = {
            "its": (3, 32, 32),
            "femnist_iid": (1, 28, 28),
            "femnist_niid": (1, 28, 28),
            "flair": (3, 224, 224),
            "pneumoniamnist": (1, 28, 28),
            "bloodmnist": (3, 28, 28),
            "lung_photos": (3, 224, 224),
            "mock": (3, 32, 32),
        }

        input_shape = input_shapes.get(self.dataset_type, (3, 32, 32))

        self.federated_dataset = MockFederatedDataset(
            num_clients=num_clients, input_shape=input_shape
        )
        self.is_setup = True

    def teardown_dataset(self) -> None:
        """Mock dataset cleanup without file operations."""
        self.federated_dataset = None
        self.is_setup = False

    def get_client_data(self, client_id: int) -> MockDataset:
        """Get data for a specific client."""
        if not self.is_setup:
            raise RuntimeError("Dataset not setup. Call setup_dataset() first.")
        return self.federated_dataset.get_client_dataset(client_id)


def generate_mock_dataset_config() -> Dict[str, str]:
    """Generate mock dataset configuration mapping."""
    return {
        "its": "datasets/its",
        "femnist_iid": "datasets/femnist_iid",
        "femnist_niid": "datasets/femnist_niid",
        "flair": "datasets/flair",
        "pneumoniamnist": "datasets/pneumoniamnist",
        "bloodmnist": "datasets/bloodmnist",
        "lung_photos": "datasets/lung_photos",
    }


def generate_mock_client_parameters(
    num_clients: int, param_size: int = 1000
) -> List[np.ndarray]:
    """
    Generate mock client parameters for testing aggregation strategies.

    Args:
        num_clients: Number of clients to generate parameters for
        param_size: Size of parameter vector for each client

    Returns:
        List of numpy arrays representing client parameters
    """
    np.random.seed(42)
    return [np.random.randn(param_size) for _ in range(num_clients)]


def generate_mock_client_metrics(
    num_clients: int, num_rounds: int
) -> Dict[int, Dict[str, List[float]]]:
    """
    Generate mock client metrics for testing history tracking.

    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds

    Returns:
        Dictionary mapping client IDs to their metrics history
    """
    np.random.seed(42)
    return {
        client_id: {
            "loss": np.random.uniform(0.1, 2.0, num_rounds).tolist(),
            "accuracy": np.random.uniform(0.5, 0.95, num_rounds).tolist(),
            "f1_score": np.random.uniform(0.4, 0.9, num_rounds).tolist(),
        }
        for client_id in range(num_clients)
    }


def generate_byzantine_client_parameters(
    num_clients: int,
    num_byzantine: int,
    param_size: int = 1000,
    attack_type: str = "gaussian",
) -> List[np.ndarray]:
    """
    Generate client parameters with Byzantine (malicious) clients for testing defense mechanisms.

    Args:
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine clients
        param_size: Size of parameter vector
        attack_type: Type of attack ("gaussian", "zero", "flip")

    Returns:
        List of parameters with Byzantine clients included
    """
    np.random.seed(42)

    # Generate honest client parameters
    honest_params = [
        np.random.randn(param_size) for _ in range(num_clients - num_byzantine)
    ]

    # Generate Byzantine client parameters based on attack type
    byzantine_params = []
    for _ in range(num_byzantine):
        if attack_type == "gaussian":
            # Add large Gaussian noise
            byzantine_params.append(np.random.randn(param_size) * 10)
        elif attack_type == "zero":
            # Send zero parameters
            byzantine_params.append(np.zeros(param_size))
        elif attack_type == "flip":
            # Flip sign of honest parameters
            base_param = np.random.randn(param_size)
            byzantine_params.append(-base_param * 5)
        else:
            # Default to Gaussian attack
            byzantine_params.append(np.random.randn(param_size) * 10)

    # Combine and shuffle
    all_params = honest_params + byzantine_params
    np.random.shuffle(all_params)

    return all_params
