"""
Global pytest configuration and fixtures for federated learning simulation tests.

This module contains only pytest-specific fixtures and configuration.
General utilities, imports, and FL helpers are in tests.common module.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
from tests.common import STRATEGY_CONFIGS, np, generate_mock_client_data
from src.data_models.simulation_strategy_history import SimulationStrategyHistory

# Deterministic test environment
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Single-threaded
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP


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


@pytest.fixture(autouse=True, scope="function")
def prevent_real_output_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent tests from creating directories in real out/ directory.

    Patches DirectoryHandler class variables to use tmp_path for all tests.
    Auto-cleanup happens via pytest's tmp_path fixture.
    """
    test_output = tmp_path / "test_output"
    test_output.mkdir()
    csv_dir = test_output / "csv"
    csv_dir.mkdir()

    monkeypatch.setattr(
        "src.output_handlers.directory_handler.DirectoryHandler.dirname",
        str(test_output),
    )
    monkeypatch.setattr(
        "src.output_handlers.directory_handler.DirectoryHandler.new_csv_dirname",
        str(csv_dir),
    )


@pytest.fixture
def mock_strategy_history():
    """Create mock strategy history for all strategy tests."""
    return Mock(spec=SimulationStrategyHistory)


@pytest.fixture
def mock_client_results_factory():
    """Factory for creating mock client results with configurable client count."""

    def _create(num_clients: int = 5, param_shape: tuple = (10, 5)):
        return generate_mock_client_data(
            num_clients=num_clients, param_shape=param_shape
        )

    return _create


@pytest.fixture
def mock_client_results(mock_client_results_factory):
    """Default mock client results with 5 clients."""
    return mock_client_results_factory(5)


@pytest.fixture
def mock_client_results_15(mock_client_results_factory):
    """Mock client results with 15 clients (for Bulyan strategy)."""
    return mock_client_results_factory(15)


@pytest.fixture
def krum_fit_metrics_fn():
    """Provide consistent fit_metrics_aggregation_fn for Krum-based strategies."""
    return lambda x: x


@pytest.fixture(scope="module")
def mock_network_model():
    """Create mock network model for PID strategy tests. Module-scoped for performance."""
    return Mock()


@pytest.fixture
def mock_loader_factory():
    """Factory for creating mock data loaders by type (cnn/transformer)."""
    import torch

    def _create(data_type="cnn", purpose="train", batch_count=5):
        if data_type == "cnn":
            shape = (3, 32, 32)  # CNN image shape
            mock_data = [
                (torch.randn(8, *shape), torch.randint(0, 10, (8,)))
                for _ in range(batch_count)
            ]
        else:  # transformer
            mock_data = []
            for _ in range(batch_count):
                batch = {
                    "input_ids": torch.randint(0, 1000, (8, 512)),
                    "attention_mask": torch.ones(8, 512),
                    "labels": torch.randint(0, 1000, (8, 512)),
                }
                mock_data.append(batch)

        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter(mock_data))
        mock_loader.__len__ = Mock(return_value=batch_count)
        return mock_loader

    return _create


@pytest.fixture
def mock_clustering_factory():
    """Factory for creating clustering mock contexts (KMeans/MinMaxScaler)."""

    def _create(strategy_module, num_clients=5):
        module_path = f"src.simulation_strategies.{strategy_module}"

        mock_data: Dict[str, Any] = {
            "kmeans": None,
            "scaler": None,
            "kmeans_instance": None,
            "scaler_instance": None,
        }

        class ClusteringContext:
            def __enter__(self):
                self.kmeans_patch = patch(f"{module_path}.KMeans")
                self.scaler_patch = patch(f"{module_path}.MinMaxScaler")

                mock_data["kmeans"] = self.kmeans_patch.start()
                mock_data["scaler"] = self.scaler_patch.start()

                mock_data["kmeans_instance"] = Mock()
                transform_result = np.array(
                    [[0.1 * (i + 1)] for i in range(num_clients)]
                )
                mock_data["kmeans_instance"].transform.return_value = transform_result
                mock_data["kmeans"].return_value.fit.return_value = mock_data[
                    "kmeans_instance"
                ]

                mock_data["scaler_instance"] = Mock()
                mock_data["scaler_instance"].transform.return_value = transform_result
                mock_data["scaler_instance"].fit.return_value = mock_data[
                    "scaler_instance"
                ]
                mock_data["scaler"].return_value = mock_data["scaler_instance"]

                return mock_data

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.kmeans_patch.stop()
                self.scaler_patch.stop()
                return False

        return ClusteringContext()

    return _create


@pytest.fixture(scope="session")
def mock_strategy_configs() -> Dict[str, Dict[str, Any]]:
    """Strategy configurations for parameterized tests."""
    return STRATEGY_CONFIGS


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
def mock_client_parameters():
    """Generate mock client parameters."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal(100) for _ in range(5)]


# Dataset Loader Testing Fixtures
@pytest.fixture
def medquad_column_names():
    """Standard column names for MedQuAD dataset mocks."""
    return ["input_ids", "attention_mask", "answer", "token_type_ids", "question"]


@pytest.fixture
def mock_dataset_dict_chain(medquad_column_names):
    """Create configured DatasetDict mock with method chaining support."""
    from unittest.mock import Mock

    mock_dataset_dict = Mock()
    mock_train_dataset = Mock()
    mock_train_dataset.column_names = medquad_column_names

    # Configure method chaining
    mock_dataset_dict.map.return_value = mock_dataset_dict
    mock_dataset_dict.remove_columns.return_value = mock_dataset_dict
    mock_dataset_dict.__getitem__ = Mock(return_value=mock_train_dataset)
    mock_train_dataset.train_test_split.return_value = {
        "train": Mock(),
        "test": Mock(),
    }

    return mock_dataset_dict, mock_train_dataset


# Attack Snapshot Testing Fixtures
@pytest.fixture
def sample_attack_data():
    """Generate sample data tensors for attack snapshot tests."""
    from tests.common import create_sample_tensors

    data, labels = create_sample_tensors(batch_size=5)
    return data, labels, labels.clone()


@pytest.fixture
def attack_config_label_flipping():
    """Generate label flipping attack configuration."""
    from tests.common import create_attack_config

    return create_attack_config("label_flipping")


@pytest.fixture
def attack_config_gaussian_noise():
    """Generate gaussian noise attack configuration."""
    from tests.common import create_attack_config

    return create_attack_config("gaussian_noise", target_noise_snr=10.0)


@pytest.fixture
def nested_attack_config():
    """Generate nested attack configuration."""
    from tests.common import create_nested_attack_config

    return create_nested_attack_config("label_flipping")


# Test failure logging setup
failure_logger = logging.getLogger("test_failure_helper")


def _setup_failure_logger():
    """Setup the failure logger only when needed."""
    if not failure_logger.handlers:
        failure_logger.setLevel(logging.INFO)
        failure_logger.propagate = False

        # Create logs directory if it doesn't exist
        log_dir = Path("tests/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"test_failures_{timestamp}.log"

        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)

        failure_logger.addHandler(fh)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Pytest hook to access test report information.

    This hook is called for each test. We use it to detect failures
    and log helpful, context-specific messages to a separate log file.
    """

    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    # Only analyze test call phase failures
    if report.when == "call" and report.failed:
        # Setup logger only when we have a failure to log
        _setup_failure_logger()
        # --- Heuristic-based failure analysis logic --- #

        # Get exception info from the call object
        excinfo = call.excinfo
        if excinfo:
            exc_type = excinfo.type
            exc_message = str(excinfo.value)
            test_path = item.fspath.strpath

            header = f"Test Failed: {item.nodeid}"
            separator = "-" * len(header)
            failure_logger.info(separator)
            failure_logger.info(header)
            failure_logger.info(f"Exception: {exc_type.__name__}")

            # Heuristic 1: ImportError
            if issubclass(exc_type, ImportError):
                failure_logger.warning(
                    "Hint: An ImportError often means a problem with your environment."
                )
                failure_logger.warning(
                    "  - Did you forget to activate the virtual environment? (`source venv/Scripts/activate`)"
                )
                failure_logger.warning(
                    "  - Are you running pytest from the project root directory?"
                )

            # Heuristic 2: FileNotFoundError
            elif issubclass(exc_type, FileNotFoundError):
                failure_logger.warning(
                    "Hint: A FileNotFoundError suggests a missing file or incorrect path."
                )
                failure_logger.warning(
                    "  - If loading data, check that the path is correct."
                )
                failure_logger.warning(
                    "  - Are you using a temporary directory fixture (e.g., `tmp_path`) correctly?"
                )

            # Heuristic 3: PyTorch Shape Errors
            elif issubclass(exc_type, RuntimeError) and (
                "shape" in exc_message or "dimension" in exc_message
            ):
                failure_logger.warning(
                    "Hint: A RuntimeError mentioning 'shape' or 'dimension' is a common PyTorch error."
                )
                failure_logger.warning(
                    "  - Your tensor dimensions might not match. Check the model's input/output shapes."
                )
                failure_logger.warning(
                    "  - See `tests/docs/test_data_generation.md` to verify mock data shapes."
                )

            # Heuristic 4: Strategy/Aggregation Logic Errors
            elif "test_simulation_strategies" in test_path and issubclass(
                exc_type, AssertionError
            ):
                failure_logger.warning(
                    "Hint: An AssertionError in a strategy test points to an algorithmic problem."
                )
                failure_logger.warning(
                    "  - Does your aggregation logic handle this edge case correctly?"
                )
                failure_logger.warning(
                    "  - Review the core concepts in `tests/docs/fl_fundamentals.md`."
                )

            # Default message for other AssertionErrors
            elif issubclass(exc_type, AssertionError):
                failure_logger.warning(
                    "Hint: An AssertionError means a condition you expected to be true was false."
                )
                failure_logger.warning(
                    "  - Double-check the values being compared in your `assert` statement."
                )

            failure_logger.info(separator)
