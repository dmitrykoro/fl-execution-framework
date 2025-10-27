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

import pytest
from tests.common import STRATEGY_CONFIGS, np

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


# Strategy testing fixtures
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

    return create_attack_config("label_flipping", flip_fraction=0.7)


@pytest.fixture
def attack_config_gaussian_noise():
    """Generate gaussian noise attack configuration."""
    from tests.common import create_attack_config

    return create_attack_config("gaussian_noise", target_noise_snr=10.0)


@pytest.fixture
def nested_attack_config():
    """Generate nested attack configuration."""
    from tests.common import create_nested_attack_config

    return create_nested_attack_config("label_flipping", flip_fraction=0.5)


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
