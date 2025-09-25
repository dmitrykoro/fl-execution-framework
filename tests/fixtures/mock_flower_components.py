"""
Mock Flower FL components for testing without distributed execution.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

NDArray = np.ndarray
Config = Dict[str, Any]
Metrics = Dict[str, Any]
Parameters = Any
Scalar = Union[bool, bytes, float, int, str]

TENSOR_TYPE_NUMPY = "numpy.ndarray"


class MockParameters:
    """Mock implementation of flwr.common.Parameters."""

    def __init__(self, tensors: List[bytes], tensor_type: str = TENSOR_TYPE_NUMPY):
        """
        Initialize mock parameters.

        Args:
            tensors: List of serialized tensors
            tensor_type: Type of tensors
        """
        self.tensors = tensors
        self.tensor_type = tensor_type

    def __eq__(self, other: object) -> bool:
        """Check equality with another Parameters object."""
        if not isinstance(other, MockParameters):
            return False
        return self.tensors == other.tensors and self.tensor_type == other.tensor_type


class MockFitRes:
    """Mock implementation of flwr.common.FitRes."""

    def __init__(
        self,
        parameters: MockParameters,
        num_examples: int,
        metrics: Optional[Metrics] = None,
    ):
        """
        Initialize mock fit result.

        Args:
            parameters: Updated model parameters
            num_examples: Number of training examples used
            metrics: Optional training metrics
        """
        self.parameters = parameters
        self.num_examples = num_examples
        self.metrics = metrics or {}


class MockEvaluateRes:
    """Mock implementation of flwr.common.EvaluateRes."""

    def __init__(
        self, loss: float, num_examples: int, metrics: Optional[Metrics] = None
    ):
        """
        Initialize mock evaluation result.

        Args:
            loss: Evaluation loss
            num_examples: Number of evaluation examples
            metrics: Optional evaluation metrics
        """
        self.loss = loss
        self.num_examples = num_examples
        self.metrics = metrics or {}


class MockClientProxy:
    """Mock implementation of flwr.server.client_proxy.ClientProxy."""

    def __init__(self, cid: str, client_fn: Optional[Callable[..., Any]] = None):
        """
        Initialize mock client proxy.

        Args:
            cid: Client ID
            client_fn: Optional client function for creating actual client
        """
        self.cid = cid
        self.client_fn = client_fn
        self._mock_client = None

        # Training history for consistent behavior
        # Extract numeric part from client ID for seeding
        try:
            client_num = int(cid)
        except ValueError:
            # Use hash for non-numeric cid seeding
            client_num = hash(cid) % 1000
        self._rng = np.random.default_rng(42 + client_num)
        self._training_rounds = 0

    def fit(self, parameters: MockParameters, _: Config) -> MockFitRes:
        """
        Mock client training.

        Args:
            parameters: Model parameters from server
            _: Training configuration (unused)

        Returns:
            Mock fit result with updated parameters and metrics
        """
        self._training_rounds += 1

        # Add noise to simulate parameter updates
        updated_tensors = []
        for tensor_bytes in parameters.tensors:
            # Deserialize, add noise, serialize
            tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
            noise = self._rng.normal(0, 0.01, tensor.shape).astype(np.float32)
            updated_tensor = tensor + noise
            updated_tensors.append(updated_tensor.tobytes())

        updated_params = MockParameters(updated_tensors, parameters.tensor_type)

        # Training metrics
        mock_loss = self._rng.uniform(0.1, 2.0)
        mock_accuracy = self._rng.uniform(0.5, 0.95)
        num_examples = int(self._rng.integers(50, 200))

        metrics: Metrics = {
            "loss": mock_loss,
            "accuracy": mock_accuracy,
            "round": self._training_rounds,
        }

        return MockFitRes(updated_params, num_examples, metrics)

    def evaluate(self, _parameters: MockParameters, _config: Config) -> MockEvaluateRes:
        """
        Mock client evaluation.

        Args:
            _parameters: Model parameters from server (unused)
            _config: Evaluation configuration (unused)

        Returns:
            Mock evaluation result
        """
        # Evaluation metrics
        mock_loss = self._rng.uniform(0.1, 1.5)
        mock_accuracy = self._rng.uniform(0.6, 0.95)
        num_examples = int(self._rng.integers(30, 100))

        metrics: Metrics = {
            "accuracy": mock_accuracy,
            "f1_score": self._rng.uniform(0.5, 0.9),
        }

        return MockEvaluateRes(mock_loss, num_examples, metrics)


class MockServerConfig:
    """Mock implementation of flwr.server.ServerConfig."""

    def __init__(self, num_rounds: int):
        """
        Initialize mock server configuration.

        Args:
            num_rounds: Number of federated learning rounds
        """
        self.num_rounds = num_rounds


class MockNumPyClient:
    """Mock implementation of flwr.client.NumPyClient."""

    def __init__(self, client_id: int = 0):
        """
        Initialize mock NumPy client.

        Args:
            client_id: Client identifier
        """
        self.client_id = client_id
        self._rng = np.random.default_rng(42 + client_id)

    def get_parameters(self, _: Config) -> List[NDArray]:
        """
        Get client parameters.

        Args:
            _: Configuration dictionary (unused)

        Returns:
            List of parameter arrays
        """
        # Consistent seeding for deterministic results
        self._rng = np.random.default_rng(42 + self.client_id)
        return [
            self._rng.standard_normal((100, 10)).astype(np.float32),
            self._rng.standard_normal(10).astype(np.float32),
        ]

    def fit(
        self, parameters: List[NDArray], _: Config
    ) -> Tuple[List[NDArray], int, Metrics]:
        """
        Mock client training.

        Args:
            parameters: Model parameters from server
            _: Training configuration (unused)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Add noise to simulate training
        updated_params = []
        for param in parameters:
            noise = self._rng.normal(0, 0.01, param.shape).astype(param.dtype)
            updated_params.append(param + noise)

        num_examples = int(self._rng.integers(50, 200))
        metrics: Metrics = {
            "loss": self._rng.uniform(0.1, 2.0),
            "accuracy": self._rng.uniform(0.5, 0.95),
        }

        return updated_params, num_examples, metrics

    def evaluate(
        self, _parameters: List[NDArray], _config: Config
    ) -> Tuple[float, int, Metrics]:
        """
        Mock client evaluation.

        Args:
            _parameters: Model parameters from server (unused)
            _config: Evaluation configuration (unused)

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        loss = self._rng.uniform(0.1, 1.5)
        num_examples = int(self._rng.integers(30, 100))
        metrics: Metrics = {
            "accuracy": self._rng.uniform(0.6, 0.95),
            "f1_score": self._rng.uniform(0.5, 0.9),
        }

        return loss, num_examples, metrics


class MockClient:
    """Mock implementation of flwr.client.Client."""

    def __init__(self, numpy_client: MockNumPyClient):
        """
        Initialize mock client wrapper.

        Args:
            numpy_client: Underlying NumPy client
        """
        self.numpy_client = numpy_client

    def fit(self, parameters: MockParameters, config: Config) -> MockFitRes:
        """Mock fit method that delegates to NumPy client."""
        # Convert Parameters to numpy arrays
        np_params = [np.frombuffer(t, dtype=np.float32) for t in parameters.tensors]

        # Call NumPy client
        updated_params, num_examples, metrics = self.numpy_client.fit(np_params, config)

        # Convert back to Parameters
        updated_tensors = [p.tobytes() for p in updated_params]
        updated_parameters = MockParameters(updated_tensors, parameters.tensor_type)

        return MockFitRes(updated_parameters, num_examples, metrics)

    def evaluate(self, parameters: MockParameters, config: Config) -> MockEvaluateRes:
        """Mock evaluate method that delegates to NumPy client."""
        # Convert Parameters to numpy arrays
        np_params = [np.frombuffer(t, dtype=np.float32) for t in parameters.tensors]

        # Call NumPy client
        loss, num_examples, metrics = self.numpy_client.evaluate(np_params, config)

        return MockEvaluateRes(loss, num_examples, metrics)


def mock_start_simulation(
    client_fn: Callable[[str], MockClient],
    num_clients: int,
    config: MockServerConfig,
    strategy: Any,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """
    Mock flwr.simulation.start_simulation.

    Args:
        client_fn: Function to create client instances
        num_clients: Number of clients in simulation
        config: Server configuration
        strategy: Aggregation strategy
        **_kwargs: Additional simulation parameters (unused)

    Returns:
        Mock simulation results
    """

    # Initialize simulation state
    simulation_results = {
        "history": {
            "losses_distributed": [],
            "losses_centralized": [],
            "metrics_distributed": {},
            "metrics_centralized": {},
        },
        "num_rounds": config.num_rounds,
        "num_clients": num_clients,
    }

    # Create mock client proxies
    client_proxies = []
    for cid in range(num_clients):
        proxy = MockClientProxy(str(cid), client_fn)
        client_proxies.append(proxy)

    # Simulate federated learning rounds
    for round_num in range(config.num_rounds):
        round_results = _simulate_round(
            client_proxies, strategy, round_num, num_clients
        )

        # Update simulation history
        avg_loss = round_results["avg_loss"]
        simulation_results["history"]["losses_distributed"].append(avg_loss)

        # Add round metrics
        metrics_dict = round_results["metrics"]
        for metric_name, metric_value in metrics_dict.items():
            distributed_metrics = simulation_results["history"]["metrics_distributed"]
            if metric_name not in distributed_metrics:
                distributed_metrics[metric_name] = []
            distributed_metrics[metric_name].append(metric_value)

    return simulation_results


def _simulate_round(
    client_proxies: List[MockClientProxy],
    _: Any,
    round_num: int,
    num_clients: int,
) -> Dict[str, Any]:
    """
    Simulate single federated learning round.

    Args:
        client_proxies: List of client proxies
        _: Aggregation strategy (unused)
        round_num: Current round number
        num_clients: Total number of clients

    Returns:
        Round results dictionary
    """
    # Mock initial parameters for the round
    rng = np.random.default_rng(42 + round_num)
    mock_params = MockParameters(
        [rng.standard_normal(100).astype(np.float32).tobytes()], TENSOR_TYPE_NUMPY
    )

    # Simulate client selection (use all clients for simplicity)
    selected_clients = client_proxies[: min(num_clients, len(client_proxies))]

    # Simulate client training
    fit_results = []
    for client in selected_clients:
        fit_res = client.fit(mock_params, {"round": round_num})
        fit_results.append(fit_res)

    # Simulate client evaluation
    eval_results = []
    for client in selected_clients:
        eval_res = client.evaluate(mock_params, {"round": round_num})
        eval_results.append(eval_res)

    # Calculate round metrics
    avg_loss = np.mean([res.loss for res in eval_results])
    avg_accuracy = np.mean(
        [float(res.metrics.get("accuracy", 0.0)) for res in eval_results]
    )

    return {
        "avg_loss": avg_loss,
        "metrics": {"accuracy": avg_accuracy, "num_clients": len(selected_clients)},
        "fit_results": fit_results,
        "eval_results": eval_results,
    }


# Utility functions for parameter conversion (mock implementations)


def mock_ndarrays_to_parameters(ndarrays: List[NDArray]) -> MockParameters:
    """
    Mock flwr.common.ndarrays_to_parameters.

    Args:
        ndarrays: List of numpy arrays

    Returns:
        Mock Parameters object
    """
    tensors = [arr.astype(np.float32).tobytes() for arr in ndarrays]
    return MockParameters(tensors, TENSOR_TYPE_NUMPY)


def mock_parameters_to_ndarrays(parameters: MockParameters) -> List[NDArray]:
    """
    Mock flwr.common.parameters_to_ndarrays.

    Args:
        parameters: Mock Parameters object

    Returns:
        List of numpy arrays
    """
    return [np.frombuffer(tensor, dtype=np.float32) for tensor in parameters.tensors]


def mock_weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """
    Mock implementation of flwr.server.strategy.aggregate.weighted_loss_avg.

    Args:
        results: List of (num_examples, loss) tuples

    Returns:
        Weighted average loss
    """
    if not results:
        return 0.0

    total_examples = sum(num_examples for num_examples, _ in results)
    if total_examples == 0:
        return 0.0

    weighted_sum = sum(num_examples * loss for num_examples, loss in results)
    return weighted_sum / total_examples


# Factory functions for creating mock components


def create_mock_flower_client(client_id: int = 0) -> MockClient:
    """
    Create mock Flower client.

    Args:
        client_id: Client identifier

    Returns:
        Mock Flower client
    """
    numpy_client = MockNumPyClient(client_id)
    return MockClient(numpy_client)


def create_mock_client_proxies(num_clients: int) -> List[MockClientProxy]:
    """
    Create multiple mock client proxies for testing.

    Args:
        num_clients: Number of client proxies to create

    Returns:
        List of mock client proxies
    """
    return [MockClientProxy(str(i)) for i in range(num_clients)]


def create_mock_fit_results(
    num_clients: int, param_shapes: List[Tuple[int, ...]]
) -> List[MockFitRes]:
    """
    Create mock fit results for testing aggregation strategies.

    Args:
        num_clients: Number of clients
        param_shapes: Shapes of model parameters

    Returns:
        List of mock fit results
    """
    results = []

    for client_id in range(num_clients):
        rng = np.random.default_rng(42 + client_id)

        # Create mock parameters
        tensors = []
        for shape in param_shapes:
            param = rng.standard_normal(shape).astype(np.float32)
            tensors.append(param.tobytes())

        parameters = MockParameters(tensors, TENSOR_TYPE_NUMPY)
        num_examples = int(rng.integers(50, 200))
        metrics: Metrics = {
            "loss": rng.uniform(0.1, 2.0),
            "accuracy": rng.uniform(0.5, 0.95),
        }

        results.append(MockFitRes(parameters, num_examples, metrics))

    return results


def create_mock_evaluate_results(num_clients: int) -> List[MockEvaluateRes]:
    """
    Create mock evaluation results for testing.

    Args:
        num_clients: Number of clients

    Returns:
        List of mock evaluation results
    """
    results = []

    for client_id in range(num_clients):
        rng = np.random.default_rng(42 + client_id)

        loss = rng.uniform(0.1, 1.5)
        num_examples = int(rng.integers(30, 100))
        metrics: Metrics = {
            "accuracy": rng.uniform(0.6, 0.95),
            "f1_score": rng.uniform(0.5, 0.9),
        }

        results.append(MockEvaluateRes(loss, num_examples, metrics))

    return results
