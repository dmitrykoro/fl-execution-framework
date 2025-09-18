"""
Lightweight mock neural networks for federated learning test scenarios.

Provides minimal PyTorch models that simulate training behavior without
actual gradient computation for fast test execution.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

NDArray = np.ndarray
Config = Dict[str, Any]
Metrics = Dict[str, Any]


class MockBaseNetwork(nn.Module):
    """Base class for mock networks with parameter serialization methods.

    Provides standard get/set parameter methods for model state transfer
    in federated learning scenarios.
    """

    def get_parameters(self) -> List[NDArray]:
        """Extract model weights as numpy arrays.

        Returns:
            List of numpy arrays containing model parameters.
        """
        return [param.detach().numpy() for param in self.parameters()]

    def set_parameters(self, parameters: List[NDArray]) -> None:
        """Load model weights from numpy arrays.

        Args:
            parameters: List of numpy arrays to load as model weights.
        """
        params_dict = zip(self.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.as_tensor(new_param, dtype=param.dtype)


class MockNetwork(MockBaseNetwork):
    """Single linear layer network for basic federated learning tests.

    Minimal architecture with one fully-connected layer for fast execution
    in test scenarios requiring model parameter exchange.
    """

    def __init__(self, num_classes: int = 10, input_size: int = 3072):
        """
        Args:
            num_classes: Number of output classes for classification.
            input_size: Flattened input dimension (default: 3072 for CIFAR-10).
        """
        super(MockNetwork, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.fc = nn.Linear(input_size, num_classes)

        # Deterministic initialization for test reproducibility
        torch.manual_seed(42)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic input flattening.

        Args:
            x: Input tensor of any shape.

        Returns:
            Class logits tensor of shape (batch_size, num_classes).
        """
        # Auto-flatten multi-dimensional inputs
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)  # type: ignore[no-any-return]


class MockCNNNetwork(MockBaseNetwork):
    """Minimal CNN for image classification test scenarios.

    Single convolutional layer followed by adaptive pooling and
    linear classifier for fast test execution.
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Args:
            num_classes: Number of output classes for classification.
            input_channels: Input channels (1=grayscale, 3=RGB).
        """
        super(MockCNNNetwork, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Minimal architecture for test speed
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

        # Deterministic initialization for test reproducibility
        torch.manual_seed(42)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Xavier initialization to all layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv -> relu -> pool -> flatten -> linear.

        Args:
            x: Input image tensor.

        Returns:
            Class logits tensor.
        """
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MockFlowerClient:
    """Simulated federated learning client for test scenarios.

    Implements fit() and evaluate() methods that simulate training
    by adding noise to parameters instead of actual gradient computation.
    """

    def __init__(self, client_id: int, model: nn.Module, dataset_size: int = 100):
        """
        Args:
            client_id: Unique client identifier.
            model: PyTorch model instance.
            dataset_size: Number of samples in client's dataset.
        """
        self.client_id = client_id
        self.model: MockBaseNetwork = model  # type: ignore[assignment]
        self.dataset_size = dataset_size

        # Track simulated training metrics
        self.training_history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    def fit(
        self, parameters: List[NDArray], config: Config
    ) -> Tuple[List[NDArray], int, Metrics]:
        """
        Simulate training by adding noise to parameters.

        Args:
            parameters: Model weights from server.
            config: Training configuration dict.

        Returns:
            Tuple of (updated_weights, dataset_size, training_metrics).
        """
        self.model.set_parameters(parameters)

        # Simulate training with parameter noise instead of gradients
        np.random.seed(42 + self.client_id)
        updated_params = []
        for param in parameters:
            noise = np.random.normal(0, 0.01, param.shape)
            updated_params.append(param + noise)

        # Generate random training metrics
        mock_loss = np.random.uniform(0.1, 2.0)
        mock_accuracy = np.random.uniform(0.5, 0.95)

        self.training_history["loss"].append(mock_loss)
        self.training_history["accuracy"].append(mock_accuracy)

        metrics: Metrics = {"loss": mock_loss, "accuracy": mock_accuracy}

        return updated_params, self.dataset_size, metrics

    def evaluate(
        self, parameters: List[NDArray], config: Config
    ) -> Tuple[float, int, Metrics]:
        """
        Simulate model evaluation with random metrics.

        Args:
            parameters: Model weights from server.
            config: Evaluation configuration dict.

        Returns:
            Tuple of (loss, dataset_size, evaluation_metrics).
        """
        self.model.set_parameters(parameters)

        # Generate random evaluation metrics
        np.random.seed(42 + self.client_id)
        mock_loss = np.random.uniform(0.1, 1.5)
        mock_accuracy = np.random.uniform(0.6, 0.95)
        mock_f1 = np.random.uniform(0.5, 0.9)

        metrics: Metrics = {"accuracy": mock_accuracy, "f1_score": mock_f1}

        return mock_loss, self.dataset_size, metrics


class MockNetworkFactory:
    """Factory for creating dataset-appropriate mock networks.

    Maps dataset types to suitable network architectures (CNN vs linear)
    based on data characteristics.
    """

    @staticmethod
    def create_network(dataset_type: str, num_classes: int = 10) -> nn.Module:
        """
        Create network architecture based on dataset characteristics.

        Args:
            dataset_type: Dataset identifier (e.g., 'femnist_iid', 'medquad').
            num_classes: Number of output classes.

        Returns:
            MockCNNNetwork for image data, MockNetwork for text data.
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
            input_channels = config.get("input_channels", 3)
            if not isinstance(input_channels, int):
                input_channels = 3
            return MockCNNNetwork(
                num_classes=num_classes, input_channels=input_channels
            )
        else:
            input_size = config.get("input_size", 3072)
            if not isinstance(input_size, int):
                input_size = 3072
            return MockNetwork(num_classes=num_classes, input_size=input_size)


def create_mock_client_models(
    num_clients: int, dataset_type: str = "its", num_classes: int = 10
) -> List[MockFlowerClient]:
    """
    Generate list of federated learning clients with varying dataset sizes.

    Args:
        num_clients: Number of clients to create.
        dataset_type: Dataset identifier for network selection.
        num_classes: Number of output classes.

    Returns:
        List of MockFlowerClient instances with random dataset sizes.
    """
    clients = []

    for client_id in range(num_clients):
        model = MockNetworkFactory.create_network(dataset_type, num_classes)

        # Assign random dataset size for heterogeneous simulation
        np.random.seed(42 + client_id)
        dataset_size = np.random.randint(50, 200)

        client = MockFlowerClient(client_id, model, dataset_size)
        clients.append(client)

    return clients


def generate_mock_model_parameters(model: nn.Module) -> List[NDArray]:
    """
    Create random parameters matching model architecture.

    Args:
        model: PyTorch model to extract shapes from.

    Returns:
        List of numpy arrays with matching parameter shapes.
    """
    torch.manual_seed(42)
    mock_params = []

    for param in model.parameters():
        # Create random weights with matching dimensions
        mock_param = torch.randn_like(param).detach().numpy()
        mock_params.append(mock_param)

    return mock_params


def create_mock_aggregated_parameters(
    client_parameters: List[List[NDArray]], weights: Optional[List[float]] = None
) -> List[NDArray]:
    """
    Aggregate client parameters using weighted averaging.

    Args:
        client_parameters: Parameter lists from each client.
        weights: Optional weights for each client (default: equal weighting).

    Returns:
        Aggregated parameter list representing global model state.
    """
    if not client_parameters:
        raise ValueError("No client parameters provided")

    num_clients = len(client_parameters)

    weights_array: NDArray
    if weights is None:
        weights_array = np.ones(num_clients) / num_clients
    else:
        weights_array = np.array(weights)

    # Use numpy vectorization for efficient aggregation
    aggregated = []
    for param_idx in range(len(client_parameters[0])):
        param_stack = np.stack(
            [client_params[param_idx] for client_params in client_parameters]
        )
        weighted_avg = np.average(param_stack, axis=0, weights=weights_array)
        aggregated.append(weighted_avg)

    return aggregated
