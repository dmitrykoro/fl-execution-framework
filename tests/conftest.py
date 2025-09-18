"""
Global pytest configuration and fixtures for federated learning simulation tests.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from flwr.common import FitRes, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

# Deterministic test environment
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Single-threaded
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP

# Type definitions
NDArray = np.ndarray
Config = Dict[str, Any]
Metrics = Dict[str, Any]
Parameters = Any


# Output directory fixture
@pytest.fixture
def mock_output_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create test output directory structure and mock DirectoryHandler."""
    output_dir = tmp_path / "out" / "test_run"
    output_dir.mkdir(parents=True)
    (output_dir / "output.log").touch()

    # Mock DirectoryHandler.dirname for test directory
    monkeypatch.setattr(
        "src.output_handlers.directory_handler.DirectoryHandler.dirname",
        str(output_dir),
    )

    return output_dir


# Strategy testing fixtures
@pytest.fixture
def mock_strategy_configs() -> Dict[str, Dict[str, Any]]:
    """Strategy configurations for parameterized tests."""
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
        "trimmed_mean": {
            "aggregation_strategy_keyword": "trimmed_mean",
            "num_of_rounds": 4,
            "num_of_clients": 10,
            "trim_ratio": 0.2,
        },
    }


@pytest.fixture(params=["trust", "pid", "krum", "multi-krum", "trimmed_mean"])
def strategy_config(
    request: pytest.FixtureRequest, mock_strategy_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Parameterized strategy configurations."""
    return mock_strategy_configs[request.param]


@pytest.fixture(params=["its", "femnist_iid", "pneumoniamnist", "bloodmnist"])
def dataset_type(request: pytest.FixtureRequest) -> str:
    """Parameterized dataset types."""
    return str(request.param)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create temporary dataset directory."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("mock training data")
    (dataset_dir / "test.txt").write_text("mock test data")
    return dataset_dir


@pytest.fixture
def mock_client_parameters() -> List[NDArray]:
    """Generate mock client parameters."""
    np.random.seed(42)
    return [np.random.randn(100) for _ in range(5)]


def generate_mock_client_data(
    num_clients: int, param_shape: Tuple[int, int] = (10, 5)
) -> List[Tuple[ClientProxy, FitRes]]:
    """Generate mock client results (ClientProxy, FitRes)."""
    results: List[Tuple[ClientProxy, FitRes]] = []
    np.random.seed(42)

    for i in range(num_clients):
        client_proxy = Mock(spec=ClientProxy)
        client_proxy.cid = str(i)

        # Create varied mock parameters
        if i < 2:  # Similar parameters for first two clients
            mock_params = [
                np.random.randn(*param_shape) * 0.1,
                np.random.randn(param_shape[1]) * 0.1,
            ]
        else:  # Different parameters for remaining clients
            mock_params = [
                np.random.randn(*param_shape) * (i + 1),
                np.random.randn(param_shape[1]) * (i + 1),
            ]

        fit_res = Mock(spec=FitRes)
        fit_res.parameters = ndarrays_to_parameters(mock_params)
        fit_res.num_examples = 100
        fit_res.metrics = {"accuracy": 0.8 + i * 0.01, "loss": 0.5 - i * 0.02}

        results.append((client_proxy, fit_res))

    return results
