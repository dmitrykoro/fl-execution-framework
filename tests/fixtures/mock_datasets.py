"""
Mock dataset generators for federated learning tests.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NDArray = np.ndarray


class MockDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Lightweight mock dataset without real data dependencies."""

    def __init__(
        self,
        size: int = 100,
        num_classes: int = 10,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        use_default_seed: bool = True,
    ):
        """
        Initialize mock dataset with specified parameters.

        Args:
            size: Number of samples in the dataset
            num_classes: Number of classification classes
            input_shape: Shape of input data (channels, height, width)
            use_default_seed: Whether to set default seeds
        """
        self.size = size
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Generate reproducible data
        if use_default_seed:
            torch.manual_seed(42)

        # Create data and labels
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class MockFederatedDataset:
    """Mock federated dataset with client data distribution."""

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
        """Generate datasets for each client."""
        client_datasets = {}

        for client_id in range(self.num_clients):
            # Different seeds simulate data heterogeneity
            torch.manual_seed(42 + client_id)

            client_datasets[client_id] = MockDataset(
                size=self.samples_per_client,
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                use_default_seed=False,
            )

        return client_datasets

    def get_client_dataset(self, client_id: int) -> MockDataset:
        """Get dataset for a specific client."""
        if client_id not in self.client_datasets:
            raise ValueError(f"Client {client_id} not found")
        return self.client_datasets[client_id]

    def get_client_dataloader(
        self, client_id: int, batch_size: int = 32
    ) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        """Get DataLoader for a specific client."""
        dataset = self.get_client_dataset(client_id)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


class MockDatasetHandler:
    """Mock dataset handler for setup and teardown operations."""

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
        self.poisoned_client_ids: set[int] = set()

    def setup_dataset(self, num_clients: int = 10) -> None:
        """Setup dataset without file operations."""
        # Setup based on dataset type
        input_shapes = {
            "its": (3, 224, 224),
            "femnist_iid": (1, 28, 28),
            "femnist_niid": (1, 28, 28),
            "flair": (3, 224, 224),
            "pneumoniamnist": (1, 28, 28),
            "bloodmnist": (3, 28, 28),
            "lung_photos": (1, 224, 224),
            "mock": (3, 32, 32),
        }

        input_shape = input_shapes.get(self.dataset_type, (3, 32, 32))

        self.federated_dataset = MockFederatedDataset(
            num_clients=num_clients, input_shape=input_shape
        )
        self.is_setup = True

    def teardown_dataset(self) -> None:
        """Cleanup dataset without file operations."""
        self.federated_dataset = None
        self.is_setup = False

    def get_client_data(self, client_id: int) -> MockDataset:
        """Get data for a specific client."""
        if not self.is_setup or self.federated_dataset is None:
            raise RuntimeError("Dataset not setup. Call setup_dataset() first.")
        return self.federated_dataset.get_client_dataset(client_id)


def generate_mock_dataset_config() -> Dict[str, str]:
    """Generate dataset configuration mapping."""
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
) -> List[NDArray]:
    """Generate client parameters for aggregation strategy testing.

    Args:
        num_clients: Number of clients to generate parameters for
        param_size: Size of parameter vector for each client

    Returns:
        List of numpy arrays representing client parameters
    """
    rng = np.random.default_rng(42)
    return [rng.standard_normal(param_size) for _ in range(num_clients)]


def generate_mock_client_metrics(
    num_clients: int, num_rounds: int
) -> Dict[int, Dict[str, List[float]]]:
    """Generate client metrics for history tracking tests.

    Args:
        num_clients: Number of clients
        num_rounds: Number of training rounds

    Returns:
        Dictionary mapping client IDs to their metrics history
    """
    rng = np.random.default_rng(42)
    return {
        client_id: {
            "loss": rng.uniform(0.1, 2.0, num_rounds).tolist(),
            "accuracy": rng.uniform(0.5, 0.95, num_rounds).tolist(),
            "f1_score": rng.uniform(0.4, 0.9, num_rounds).tolist(),
        }
        for client_id in range(num_clients)
    }


def _create_gaussian_attack(param_size: int) -> NDArray:
    """Generate parameters with large Gaussian noise."""
    rng = np.random.default_rng(123)
    return rng.standard_normal(param_size) * 10


def _create_model_poisoning_attack(param_size: int) -> NDArray:
    """Generate parameters with targeted manipulation."""
    rng = np.random.default_rng(124)
    poisoned_param = rng.standard_normal(param_size) * 0.1
    poison_indices = rng.choice(param_size, size=param_size // 10, replace=False)
    poisoned_param[poison_indices] *= 50
    return poisoned_param


def _create_byzantine_clients_attack(param_size: int) -> NDArray:
    """Generate parameters with adversarial behavior."""
    rng = np.random.default_rng(125)
    return rng.standard_normal(param_size) * 15


def _create_gradient_inversion_attack(param_size: int) -> NDArray:
    """Generate parameters with scaled values."""
    rng = np.random.default_rng(126)
    return rng.standard_normal(param_size) * 3


def _create_label_flipping_attack(param_size: int) -> NDArray:
    """Generate parameters simulating label flipping effects."""
    rng = np.random.default_rng(127)
    flipped_param = rng.standard_normal(param_size) * 0.1
    flip_indices = rng.choice(param_size, size=param_size // 5, replace=False)
    flipped_param[flip_indices] *= -10
    return flipped_param


def _create_backdoor_attack(param_size: int) -> NDArray:
    """Generate parameters with a backdoor pattern."""
    rng = np.random.default_rng(128)
    backdoor_param = rng.standard_normal(param_size) * 0.1
    pattern_indices = np.arange(0, min(100, param_size), 10)
    backdoor_param[pattern_indices] = 5.0
    return backdoor_param


def _create_zero_attack(param_size: int) -> NDArray:
    """Generate zero parameters."""
    return np.zeros(param_size)


def _create_flip_attack(param_size: int) -> NDArray:
    """Generate parameters with flipped signs."""
    rng = np.random.default_rng(129)
    base_param = rng.standard_normal(param_size)
    return -base_param * 5


ATTACK_FUNCTIONS: Dict[str, Callable[[int], NDArray]] = {
    "gaussian_noise": _create_gaussian_attack,
    "gaussian": _create_gaussian_attack,
    "model_poisoning": _create_model_poisoning_attack,
    "byzantine_clients": _create_byzantine_clients_attack,
    "gradient_inversion": _create_gradient_inversion_attack,
    "label_flipping": _create_label_flipping_attack,
    "backdoor_attack": _create_backdoor_attack,
    "zero": _create_zero_attack,
    "flip": _create_flip_attack,
}


def generate_byzantine_client_parameters(
    num_clients: int,
    num_byzantine: int,
    param_size: int = 1000,
    attack_type: str = "gaussian",
) -> List[NDArray]:
    """Generate client parameters with Byzantine clients for defense testing.

    Args:
        num_clients: Total number of clients
        num_byzantine: Number of Byzantine clients
        param_size: Size of parameter vector
        attack_type: Type of attack (e.g., "gaussian", "model_poisoning")

    Returns:
        List of parameters with Byzantine clients included
    """
    rng = np.random.default_rng(42)

    # Generate honest parameters
    honest_params = [
        rng.standard_normal(param_size) * 0.1
        for _ in range(num_clients - num_byzantine)
    ]

    # Get attack function with default fallback
    attack_fn = ATTACK_FUNCTIONS.get(attack_type, _create_gaussian_attack)
    byzantine_params = [attack_fn(param_size) for _ in range(num_byzantine)]

    # Combine and shuffle
    all_params = honest_params + byzantine_params
    rng.shuffle(all_params)

    return all_params
