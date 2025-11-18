"""
Pytest configuration and fixtures for API unit tests.

Provides:
- api_client: FastAPI TestClient fixture
- mock_simulation_dir: Mock simulation directory structure
"""

import json
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def api_client() -> TestClient:
    """Create a FastAPI TestClient for the app."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="httpx")
        return TestClient(app)


@pytest.fixture
def mock_simulation_dir(tmp_path: Path) -> Path:
    """
    Create a mock simulation directory structure for testing.

    Creates a simulation directory with:
    - config.json with valid simulation configuration
    - metrics.csv with sample metrics data
    - plot_data_0.json with sample plot data
    - accuracy_plot.pdf (empty file for testing)
    """
    sim_dir = tmp_path / "api_run_20250107_120000"
    sim_dir.mkdir(parents=True)

    config = {
        "shared_settings": {
            "aggregation_strategy_keyword": "fedavg",
            "num_of_rounds": 5,
            "num_of_clients": 3,
            "dataset_keyword": "bloodmnist",
            "fraction_fit": 1.0,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "batch_size": 32,
        },
        "simulation_strategies": [
            {"strategy_name": "fedavg", "num_malicious_clients": 0}
        ],
    }
    (sim_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics_csv = """round,accuracy,loss
    1,0.85,0.6
    2,0.90,0.4
    """
    (sim_dir / "metrics.csv").write_text(metrics_csv)

    plot_data = {
        "rounds": [1, 2, 3],
        "accuracy": [0.75, 0.80, 0.85],
        "loss": [0.8, 0.6, 0.4],
    }
    (sim_dir / "plot_data_0.json").write_text(json.dumps(plot_data, indent=2))

    (sim_dir / "accuracy_plot.pdf").write_bytes(b"%PDF-1.4\n%mock pdf content")

    return sim_dir
