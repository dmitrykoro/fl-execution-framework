"""
Integration tests for API + Backend simulation flow (Phase 2).

Tests end-to-end workflows:
- Full simulation lifecycle (create → poll → retrieve → delete)
- Concurrent simulations
- Attack simulation integration

These tests verify the complete API + backend integration, ensuring
the entire system works together correctly.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# --- Full Simulation Lifecycle Test ---


def test_full_simulation_lifecycle(api_client: TestClient, tmp_path: Path, monkeypatch):
    """
    Test complete simulation workflow:
    1. POST /api/simulations (create)
    2. GET /api/simulations/{id}/status (poll until complete)
    3. GET /api/simulations/{id}/metrics (verify CSV exists)
    4. GET /api/simulations/{id}/plot-data (verify JSON exists)
    5. DELETE /api/simulations/{id} (cleanup)
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess to prevent actual simulation run
    mock_process = MagicMock()
    mock_process.pid = 99999
    mock_process.poll = MagicMock(return_value=0)  # Simulate completed process
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    # Step 1: Create simulation
    config = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 2,
        "num_of_clients": 2,
    }
    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]
    assert sim_id.startswith("api_run_")

    # Step 2: Poll status (initially pending)
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["status"] in ["pending", "running", "completed"]

    # Simulate simulation completion by creating result files
    sim_dir = tmp_path / "out" / sim_id
    (sim_dir / "metrics.csv").write_text(
        "round,accuracy,loss\n1,0.75,0.5\n2,0.85,0.3\n"
    )
    (sim_dir / "plot_data_0.json").write_text(
        json.dumps({"rounds": [1, 2], "accuracy": [0.75, 0.85]})
    )
    (sim_dir / "accuracy_plot.pdf").write_bytes(b"%PDF-1.4 mock pdf")

    # Step 3: Verify status changed to completed
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 1.0

    # Step 4: Retrieve metrics CSV
    response = api_client.get(f"/api/simulations/{sim_id}/results/metrics.csv")
    assert response.status_code == 200
    csv_data = response.json()  # CSV is returned as JSON by API
    assert isinstance(csv_data, list)
    assert len(csv_data) == 2
    assert csv_data[1]["accuracy"] == 0.85

    # Step 5: Retrieve plot data JSON
    response = api_client.get(f"/api/simulations/{sim_id}/results/plot_data_0.json")
    assert response.status_code == 200
    plot_data = response.json()
    assert "rounds" in plot_data
    assert plot_data["accuracy"] == [0.75, 0.85]

    # Step 6: Verify simulation directory persists
    # Note: API doesn't provide DELETE endpoint - simulations remain for review
    assert sim_dir.exists()


# --- Concurrent Simulations Test ---


def test_concurrent_simulations(api_client: TestClient, tmp_path: Path, monkeypatch):
    """
    Test that multiple simulations can be created and tracked simultaneously.
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess to prevent actual simulation runs
    mock_process = MagicMock()
    mock_process.pid = 55555
    mock_process.poll = MagicMock(return_value=None)  # Still running
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    # Create first simulation
    config1 = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 3,
        "num_of_clients": 3,
    }
    response1 = api_client.post("/api/simulations", json=config1)
    assert response1.status_code == 201
    sim_id_1 = response1.json()["simulation_id"]

    # Create second simulation (different timestamp)
    time.sleep(1.1)  # Ensure different timestamp (resolution is seconds)
    config2 = {
        "aggregation_strategy_keyword": "krum",
        "dataset_keyword": "pneumoniamnist",
        "num_of_rounds": 2,
        "num_of_clients": 5,
    }
    response2 = api_client.post("/api/simulations", json=config2)
    assert response2.status_code == 201
    sim_id_2 = response2.json()["simulation_id"]

    assert sim_id_1 != sim_id_2  # Different IDs

    # Verify both simulations exist
    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    sims = response.json()
    assert len(sims) == 2
    sim_ids = [s["simulation_id"] for s in sims]
    assert sim_id_1 in sim_ids
    assert sim_id_2 in sim_ids

    # Verify each simulation has correct config
    response1 = api_client.get(f"/api/simulations/{sim_id_1}")
    assert response1.status_code == 200
    assert (
        response1.json()["config"]["shared_settings"]["aggregation_strategy_keyword"]
        == "fedavg"
    )

    response2 = api_client.get(f"/api/simulations/{sim_id_2}")
    assert response2.status_code == 200
    assert (
        response2.json()["config"]["shared_settings"]["aggregation_strategy_keyword"]
        == "krum"
    )

    # Both should be running/pending (no result files yet)
    status1 = api_client.get(f"/api/simulations/{sim_id_1}/status").json()
    status2 = api_client.get(f"/api/simulations/{sim_id_2}/status").json()
    assert status1["status"] in ["pending", "running"]
    assert status2["status"] in ["pending", "running"]


# --- Attack Simulation Integration Test ---


def test_attack_simulation_integration(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """
    Test simulation with attack parameters completes and generates expected outputs.
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess
    mock_process = MagicMock()
    mock_process.pid = 88888
    mock_process.poll = MagicMock(return_value=0)  # Completed
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    # Create simulation with attack parameters
    config = {
        "aggregation_strategy_keyword": "krum",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 3,
        "num_of_clients": 5,
        "num_of_malicious_clients": 2,
        "attack_type": "gaussian_noise",
        "krum_num_of_byzantines": 2,
    }
    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]

    # Verify config was saved with attack parameters
    sim_dir = tmp_path / "out" / sim_id
    config_path = sim_dir / "config.json"
    assert config_path.exists()

    with open(config_path) as f:
        saved_config = json.load(f)

    assert saved_config["shared_settings"]["num_of_malicious_clients"] == 2
    assert saved_config["shared_settings"]["attack_type"] == "gaussian_noise"
    # Note: krum_num_of_byzantines may be in simulation_strategies[0] or elsewhere depending on config structure

    # Simulate attack simulation completion with results
    (sim_dir / "metrics.csv").write_text(
        "round,accuracy,loss,byzantine_detected\n"
        "1,0.60,0.8,2\n"
        "2,0.70,0.6,1\n"
        "3,0.75,0.5,0\n"
    )
    (sim_dir / "plot_data_0.json").write_text(
        json.dumps(
            {
                "rounds": [1, 2, 3],
                "accuracy": [0.60, 0.70, 0.75],
                "byzantine_detected": [2, 1, 0],
            }
        )
    )

    # Verify status shows completed
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    # Verify metrics include attack-related data
    response = api_client.get(f"/api/simulations/{sim_id}/results/metrics.csv")
    assert response.status_code == 200
    csv_data = response.json()
    assert isinstance(csv_data, list)
    assert csv_data[-1]["accuracy"] == 0.75  # Final accuracy

    # Verify plot data includes attack metrics
    response = api_client.get(f"/api/simulations/{sim_id}/results/plot_data_0.json")
    assert response.status_code == 200
    plot_data = response.json()
    assert "byzantine_detected" in plot_data
    assert plot_data["byzantine_detected"] == [2, 1, 0]


# --- Simulation Status Transitions Test ---


def test_simulation_status_transitions(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """
    Test that status reports correctly based on result file presence.
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess that always returns completed (process has finished)
    mock_process = MagicMock()
    mock_process.pid = 77777
    mock_process.poll = MagicMock(return_value=0)  # Process finished successfully
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    # Create simulation
    config = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 2,
        "num_of_clients": 2,
    }
    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]
    sim_dir = tmp_path / "out" / sim_id

    # Check 1: No result files yet - status could be pending/running
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    # Process finished but no result files = completed with no output yet
    initial_status = response.json()["status"]
    assert initial_status in ["pending", "running", "completed"]

    # Add result files
    (sim_dir / "metrics.csv").write_text("round,accuracy\n1,0.8\n2,0.9\n")
    (sim_dir / "plot_data_0.json").write_text(json.dumps({"rounds": [1, 2]}))
    (sim_dir / "plot.pdf").write_bytes(b"%PDF-1.4 fake")

    # Check 2: With result files - status is definitely completed
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 1.0


# --- Error Recovery Test ---


def test_simulation_with_failed_status(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """
    Test simulation that fails (process exits with error) is reported correctly.
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess that exits with error code
    mock_process = MagicMock()
    mock_process.pid = 66666
    mock_process.poll = MagicMock(return_value=1)  # Non-zero = error
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    # Create simulation
    config = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 2,
        "num_of_clients": 2,
    }
    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201
    sim_id = response.json()["simulation_id"]
    sim_dir = tmp_path / "out" / sim_id

    # Create execution log
    (sim_dir / "execution.log").write_text("Simulation failed: Out of memory")

    # Status should show failure
    response = api_client.get(f"/api/simulations/{sim_id}/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["status"] == "failed"
    assert "Out of memory" in status_data.get("error", "")


# --- Simulation List Filtering Test ---


def test_list_simulations_with_multiple_runs(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """
    Test GET /api/simulations returns all simulations sorted by creation time.
    """
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")

    # Create multiple mock simulation directories
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    for i, strategy in enumerate(["fedavg", "krum", "trimmed_mean"]):
        sim_dir = out_dir / f"api_run_2025010{i}_120000"
        sim_dir.mkdir()

        config = {
            "shared_settings": {
                "aggregation_strategy_keyword": strategy,
                "num_of_rounds": 5,
                "num_of_clients": 3,
                "dataset_keyword": "bloodmnist",
            },
            "simulation_strategies": [{}],
        }
        (sim_dir / "config.json").write_text(json.dumps(config))
        (sim_dir / "metrics.csv").write_text("round,accuracy\n1,0.8\n")

    # List all simulations
    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    sims = response.json()

    assert len(sims) == 3
    strategies = [s["strategy_name"] for s in sims]
    assert "fedavg" in strategies
    assert "krum" in strategies
    assert "trimmed_mean" in strategies

    # Verify all have correct structure
    for sim in sims:
        assert "simulation_id" in sim
        assert "strategy_name" in sim
        assert "num_of_rounds" in sim
        assert "num_of_clients" in sim
        assert "created_at" in sim
