"""
Global pytest configuration and fixtures for federated learning simulation tests.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from tests.fixtures.mock_datasets import (
    MockDataset,
    MockDatasetHandler,
    MockFederatedDataset,
    generate_byzantine_client_parameters,
)
from tests.fixtures.mock_flower_components import (
    MockServerConfig,
    create_mock_client_proxies,
    create_mock_evaluate_results,
    create_mock_fit_results,
    create_mock_flower_client,
    mock_start_simulation,
)
from tests.fixtures.sample_models import (
    MockCNNNetwork,
    MockFlowerClient,
    MockNetwork,
    create_mock_client_models,
    generate_mock_model_parameters,
)

# Configure environment for deterministic test execution
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Single-threaded to avoid subprocess issues
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads


@pytest.fixture
def sample_dataset_config() -> Dict[str, str]:
    """Sample dataset configuration mapping."""
    return {
        "its": "datasets/its",
        "femnist_iid": "datasets/femnist_iid",
        "femnist_niid": "datasets/femnist_niid",
        "flair": "datasets/flair",
        "pneumoniamnist": "datasets/pneumoniamnist",
        "bloodmnist": "datasets/bloodmnist",
        "lung_photos": "datasets/lung_photos",
    }


@pytest.fixture
def mock_client_parameters() -> List[np.ndarray]:
    """Generate mock client parameters for testing."""
    np.random.seed(42)
    return [np.random.randn(100) for _ in range(5)]


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory for testing."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()

    # Create some mock dataset files
    (dataset_dir / "train.txt").write_text("mock training data")
    (dataset_dir / "test.txt").write_text("mock test data")

    return dataset_dir


@pytest.fixture
def mock_strategy_configs() -> Dict[str, Dict[str, Any]]:
    """Collection of all strategy configurations for parameterized testing."""
    return {
        "trust": {
            "aggregation_strategy_keyword": "trust",
            "num_of_rounds": 5,
            "num_of_clients": 10,
            "trust_threshold": 0.7,
            "beta_value": 0.5,
        },
        "pid": {
            "aggregation_strategy_keyword": "pid",
            "num_of_rounds": 3,
            "num_of_clients": 8,
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
        },
        "pid_scaled": {
            "aggregation_strategy_keyword": "pid_scaled",
            "num_of_rounds": 3,
            "num_of_clients": 8,
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
        },
        "pid_standardized": {
            "aggregation_strategy_keyword": "pid_standardized",
            "num_of_rounds": 3,
            "num_of_clients": 8,
            "Kp": 1.0,
            "Ki": 0.1,
            "Kd": 0.01,
        },
        "krum": {
            "aggregation_strategy_keyword": "krum",
            "num_of_rounds": 4,
            "num_of_clients": 12,
            "num_krum_selections": 8,
        },
        "multi-krum": {
            "aggregation_strategy_keyword": "multi-krum",
            "num_of_rounds": 4,
            "num_of_clients": 12,
            "num_krum_selections": 8,
        },
        "multi-krum-based": {
            "aggregation_strategy_keyword": "multi-krum-based",
            "num_of_rounds": 4,
            "num_of_clients": 12,
            "num_krum_selections": 8,
        },
        "trimmed_mean": {
            "aggregation_strategy_keyword": "trimmed_mean",
            "num_of_rounds": 4,
            "num_of_clients": 10,
            "trim_ratio": 0.2,
        },
        "rfa": {
            "aggregation_strategy_keyword": "rfa",
            "num_of_rounds": 4,
            "num_of_clients": 10,
        },
        "bulyan": {
            "aggregation_strategy_keyword": "bulyan",
            "num_of_rounds": 4,
            "num_of_clients": 15,
        },
    }


# Test utilities
def _assert_strategy_config_valid(config: Dict[str, Any]) -> None:
    """Assert that a strategy configuration is valid."""
    assert "aggregation_strategy_keyword" in config
    assert "num_of_rounds" in config
    assert "num_of_clients" in config
    assert config["num_of_rounds"] > 0
    assert config["num_of_clients"] > 0


# Additional fixtures for mock datasets and models


@pytest.fixture
def mock_dataset() -> MockDataset:
    """Create a mock dataset for testing."""
    return MockDataset(size=100, num_classes=10)


@pytest.fixture
def mock_federated_dataset() -> MockFederatedDataset:
    """Create a mock federated dataset for testing."""
    return MockFederatedDataset(num_clients=5, samples_per_client=50)


@pytest.fixture
def mock_dataset_handler() -> MockDatasetHandler:
    """Create a mock dataset handler for testing."""
    handler = MockDatasetHandler(dataset_type="mock")
    handler.setup_dataset(num_clients=5)
    yield handler
    handler.teardown_dataset()


@pytest.fixture
def mock_network() -> MockNetwork:
    """Create a mock network for testing."""
    return MockNetwork(num_classes=10, input_size=3072)


@pytest.fixture
def mock_cnn_network() -> MockCNNNetwork:
    """Create a mock CNN network for testing."""
    return MockCNNNetwork(num_classes=10, input_channels=3)


@pytest.fixture
def mock_flower_clients() -> List[MockFlowerClient]:
    """Create mock Flower clients for testing."""
    return create_mock_client_models(num_clients=5, dataset_type="its")


@pytest.fixture
def mock_model_parameters(mock_network: MockNetwork) -> List[np.ndarray]:
    """Generate mock model parameters matching network structure."""
    return generate_mock_model_parameters(mock_network)


@pytest.fixture
def mock_byzantine_parameters() -> List[np.ndarray]:
    """Generate client parameters with Byzantine clients for testing defense mechanisms."""
    return generate_byzantine_client_parameters(
        num_clients=10, num_byzantine=3, param_size=1000, attack_type="gaussian"
    )


@pytest.fixture
def mock_output_directory(tmp_path, monkeypatch):
    """Create proper output directory structure for tests and mock DirectoryHandler."""
    output_dir = tmp_path / "out" / "test_run"
    output_dir.mkdir(parents=True)
    (output_dir / "output.log").touch()

    # Mock DirectoryHandler.dirname to point to test directory
    monkeypatch.setattr(
        "src.output_handlers.directory_handler.DirectoryHandler.dirname",
        str(output_dir),
    )

    return output_dir


@pytest.fixture(params=["trust", "pid", "krum", "multi-krum", "trimmed_mean"])
def strategy_config(
    request, mock_strategy_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Parameterized fixture for different strategy configurations."""
    return mock_strategy_configs[request.param]


@pytest.fixture(params=["its", "femnist_iid", "pneumoniamnist", "bloodmnist"])
def dataset_type(request) -> str:
    """Parameterized fixture for different dataset types."""
    return request.param


# Test utilities and helper functions


def _generate_mock_client_data(
    num_clients: int, data_size: int = 100
) -> List[np.ndarray]:
    """Generate mock client data for testing."""
    np.random.seed(42)
    return [np.random.randn(data_size) for _ in range(num_clients)]


def _assert_client_info_consistent(client_info) -> None:
    """Assert that client info maintains data consistency."""
    assert hasattr(client_info, "loss_history")
    assert hasattr(client_info, "accuracy_history")
    assert len(client_info.loss_history) == len(client_info.accuracy_history)


def _assert_parameters_shape_match(
    params1: List[np.ndarray], params2: List[np.ndarray]
) -> None:
    """Assert that two parameter lists have matching shapes."""
    assert len(params1) == len(params2)
    for p1, p2 in zip(params1, params2):
        assert p1.shape == p2.shape


@pytest.fixture
def temp_simulation_dir(tmp_path: Path) -> Path:
    """Create a complete temporary simulation directory structure."""
    sim_dir = tmp_path / "simulation"
    sim_dir.mkdir()

    # Create subdirectories
    (sim_dir / "datasets").mkdir()
    (sim_dir / "output").mkdir()
    (sim_dir / "config").mkdir()

    # Create mock dataset files
    for dataset in ["its", "femnist_iid", "pneumoniamnist"]:
        dataset_dir = sim_dir / "datasets" / dataset
        dataset_dir.mkdir()
        (dataset_dir / "train.txt").write_text(f"mock {dataset} training data")
        (dataset_dir / "test.txt").write_text(f"mock {dataset} test data")

    # Create mock config files
    config_data = {
        "its": str(sim_dir / "datasets" / "its"),
        "femnist_iid": str(sim_dir / "datasets" / "femnist_iid"),
        "pneumoniamnist": str(sim_dir / "datasets" / "pneumoniamnist"),
    }

    config_file = sim_dir / "config" / "dataset_mapping.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return sim_dir


def _create_temp_config_file(tmp_path: Path, config_data: Dict[str, Any]) -> Path:
    """Create a temporary configuration file with given data."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    return config_file


# Flower FL component fixtures


@pytest.fixture
def mock_flower_client():
    """Create a mock Flower client for testing."""
    return create_mock_flower_client(client_id=0)


@pytest.fixture
def mock_client_proxies():
    """Create mock client proxies for testing."""
    return create_mock_client_proxies(5)


@pytest.fixture
def mock_server_config():
    """Create mock server configuration for testing."""
    return MockServerConfig(num_rounds=3)


@pytest.fixture
def mock_fit_results():
    """Create mock fit results for testing aggregation."""
    param_shapes = [(100, 10), (10,)]  # Common parameter shapes
    return create_mock_fit_results(5, param_shapes)


@pytest.fixture
def mock_evaluate_results():
    """Create mock evaluation results for testing."""
    return create_mock_evaluate_results(5)


@pytest.fixture
def mock_parameters():
    """Create mock parameters for testing."""
    from tests.fixtures.mock_flower_components import mock_ndarrays_to_parameters

    arrays = [
        np.random.randn(100, 10).astype(np.float32),
        np.random.randn(10).astype(np.float32),
    ]
    return mock_ndarrays_to_parameters(arrays)


@pytest.fixture
def mock_simulation_function():
    """Create a mock simulation function for testing."""

    def client_fn(cid: str):
        return create_mock_flower_client(int(cid))

    def run_mock_simulation(num_clients=5, num_rounds=3, strategy=None):
        config = MockServerConfig(num_rounds)
        return mock_start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=config,
            strategy=strategy or MockStrategy(),
        )

    return run_mock_simulation


class MockStrategy:
    """Mock aggregation strategy for testing."""

    def __init__(self):
        self.name = "mock_strategy"

    def aggregate_fit(self, results):
        """Mock aggregate fit method."""
        return None, {}

    def aggregate_evaluate(self, results):
        """Mock aggregate evaluate method."""
        return None, {}


@pytest.fixture
def mock_client_results():
    """Create a function that generates mock client results for testing."""

    def _create_mock_client_results(num_clients: int = 5):
        """Generate mock client results with specified number of clients."""
        results = []
        np.random.seed(42)  # For reproducible tests

        for i in range(num_clients):
            client_proxy = Mock(spec=ClientProxy)
            client_proxy.cid = str(i)

            # Create mock parameters with different patterns
            if i < 2:  # First two clients have similar parameters
                mock_params = [np.random.randn(10, 5) * 0.1, np.random.randn(5) * 0.1]
            else:  # Other clients have different parameters
                mock_params = [
                    np.random.randn(10, 5) * (i + 1),
                    np.random.randn(5) * (i + 1),
                ]

            fit_res = Mock(spec=FitRes)
            fit_res.parameters = ndarrays_to_parameters(mock_params)
            fit_res.num_examples = 100
            fit_res.metrics = {"accuracy": 0.8 + i * 0.01, "loss": 0.5 - i * 0.02}

            results.append((client_proxy, fit_res))

        return results

    return _create_mock_client_results
