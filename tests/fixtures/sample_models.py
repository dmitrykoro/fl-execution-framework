"""
Mock network model classes for testing without heavy PyTorch computations.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MockNetwork(nn.Module):
    """Lightweight mock network for testing without actual training."""

    def __init__(self, num_classes: int = 10, input_size: int = 3072):
        """
        Initialize mock network.

        Args:
            num_classes: Number of output classes
            input_size: Size of flattened input (e.g., 3*32*32 = 3072 for CIFAR-10)
        """
        super(MockNetwork, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Simple linear layer for fast computation
        self.fc = nn.Linear(input_size, num_classes)

        # Initialize with reproducible weights
        torch.manual_seed(42)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [param.detach().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype)


class MockCNNNetwork(nn.Module):
    """Mock CNN network for image classification tasks."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Initialize mock CNN.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(MockCNNNetwork, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Minimal CNN architecture for fast computation
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

        # Initialize with reproducible weights
        torch.manual_seed(42)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [param.detach().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype)


class MockFlowerClient:
    """Mock Flower client for testing federated learning workflows."""

    def __init__(self, client_id: int, model: nn.Module, dataset_size: int = 100):
        """
        Initialize mock Flower client.

        Args:
            client_id: Unique client identifier
            model: PyTorch model for the client
            dataset_size: Size of client's dataset
        """
        self.client_id = client_id
        self.model = model
        self.dataset_size = dataset_size

        # Mock training history
        self.training_history = {"loss": [], "accuracy": []}

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Mock training (fit) method.

        Args:
            parameters: Model parameters from server
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, dataset_size, metrics)
        """
        # Set parameters
        self.model.set_parameters(parameters)

        # Simulate training by adding small random noise to parameters
        np.random.seed(42 + self.client_id)
        updated_params = []
        for param in parameters:
            noise = np.random.normal(0, 0.01, param.shape)
            updated_params.append(param + noise)

        # Mock training metrics
        mock_loss = np.random.uniform(0.1, 2.0)
        mock_accuracy = np.random.uniform(0.5, 0.95)

        self.training_history["loss"].append(mock_loss)
        self.training_history["accuracy"].append(mock_accuracy)

        metrics = {"loss": mock_loss, "accuracy": mock_accuracy}

        return updated_params, self.dataset_size, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Mock evaluation method.

        Args:
            parameters: Model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, dataset_size, metrics)
        """
        # Set parameters
        self.model.set_parameters(parameters)

        # Mock evaluation metrics
        np.random.seed(42 + self.client_id)
        mock_loss = np.random.uniform(0.1, 1.5)
        mock_accuracy = np.random.uniform(0.6, 0.95)
        mock_f1 = np.random.uniform(0.5, 0.9)

        metrics = {"accuracy": mock_accuracy, "f1_score": mock_f1}

        return mock_loss, self.dataset_size, metrics


class MockNetworkFactory:
    """Factory class for creating mock networks based on dataset types."""

    @staticmethod
    def create_network(dataset_type: str, num_classes: int = 10) -> nn.Module:
        """
        Create appropriate mock network for dataset type.

        Args:
            dataset_type: Type of dataset
            num_classes: Number of classes

        Returns:
            Mock network appropriate for the dataset
        """
        network_configs = {
            "its": {"input_channels": 3, "network_type": "cnn"},
            "femnist_iid": {"input_channels": 1, "network_type": "cnn"},
            "femnist_niid": {"input_channels": 1, "network_type": "cnn"},
            "flair": {"input_channels": 3, "network_type": "cnn"},
            "pneumoniamnist": {"input_channels": 1, "network_type": "cnn"},
            "bloodmnist": {"input_channels": 3, "network_type": "cnn"},
            "lung_photos": {"input_channels": 1, "network_type": "cnn"},
            "medquad": {"input_size": 768, "network_type": "linear"},  # Text data
        }

        config = network_configs.get(
            dataset_type, {"input_channels": 3, "network_type": "cnn"}
        )

        if config["network_type"] == "cnn":
            return MockCNNNetwork(
                num_classes=num_classes, input_channels=config["input_channels"]
            )
        else:
            return MockNetwork(num_classes=num_classes, input_size=config["input_size"])


def create_mock_client_models(
    num_clients: int, dataset_type: str = "its", num_classes: int = 10
) -> List[MockFlowerClient]:
    """
    Create multiple mock Flower clients with appropriate models.

    Args:
        num_clients: Number of clients to create
        dataset_type: Type of dataset
        num_classes: Number of classes

    Returns:
        List of mock Flower clients
    """
    clients = []

    for client_id in range(num_clients):
        # Create model for this client
        model = MockNetworkFactory.create_network(dataset_type, num_classes)

        # Create mock client with varying dataset sizes
        np.random.seed(42 + client_id)
        dataset_size = np.random.randint(50, 200)

        client = MockFlowerClient(client_id, model, dataset_size)
        clients.append(client)

    return clients


def generate_mock_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Generate mock model parameters with the same shape as the given model.

    Args:
        model: PyTorch model to match parameter shapes

    Returns:
        List of numpy arrays with same shapes as model parameters
    """
    torch.manual_seed(42)
    mock_params = []

    for param in model.parameters():
        # Generate random parameters with same shape
        mock_param = torch.randn_like(param).detach().numpy()
        mock_params.append(mock_param)

    return mock_params


def create_mock_aggregated_parameters(
    client_parameters: List[List[np.ndarray]], weights: Optional[List[float]] = None
) -> List[np.ndarray]:
    """
    Create mock aggregated parameters from client parameters.

    Args:
        client_parameters: List of client parameter lists
        weights: Optional weights for weighted averaging

    Returns:
        Aggregated parameters
    """
    if not client_parameters:
        raise ValueError("No client parameters provided")

    num_clients = len(client_parameters)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients

    # Initialize aggregated parameters
    aggregated = []

    for param_idx in range(len(client_parameters[0])):
        # Weighted average of parameters
        weighted_sum = np.zeros_like(client_parameters[0][param_idx])

        for client_idx, client_params in enumerate(client_parameters):
            weighted_sum += weights[client_idx] * client_params[param_idx]

        aggregated.append(weighted_sum)

    return aggregated
