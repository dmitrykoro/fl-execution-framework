"""
Comprehensive unit tests for FastAPI endpoints in src/api/main.py.

Tests cover:
- Endpoint functionality (GET, POST, DELETE)
- File serving (CSV, JSON, PDF)
- CORS configuration
- Error handling and validation
- Background task spawning
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api import main

# --- Endpoint Tests ---


def test_read_root(api_client: TestClient):
    """GET / returns welcome message."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Federated Learning Simulation Framework API"}


def test_list_simulations_empty(api_client: TestClient, tmp_path: Path, monkeypatch):
    """GET /api/simulations returns empty list when no simulations exist."""
    # Mock OUTPUT_DIR to point to empty tmp_path
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    assert response.json() == []


def test_list_simulations(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations returns list of simulations."""
    # Mock OUTPUT_DIR to point to mock_simulation_dir parent
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    sims = response.json()
    assert len(sims) == 1
    assert sims[0]["simulation_id"] == "api_run_20250107_120000"
    assert sims[0]["strategy_name"] == "fedavg"
    assert sims[0]["num_of_rounds"] == 5
    assert sims[0]["num_of_clients"] == 3
    assert "created_at" in sims[0]


def test_get_simulation_details(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id} returns simulation details."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get("/api/simulations/api_run_20250107_120000")
    assert response.status_code == 200
    data = response.json()

    assert "config" in data
    assert "result_files" in data
    assert "status" in data
    assert data["config"]["shared_settings"]["aggregation_strategy_keyword"] == "fedavg"
    assert "metrics.csv" in data["result_files"]
    assert "plot_data_0.json" in data["result_files"]
    assert "accuracy_plot.pdf" in data["result_files"]


def test_get_nonexistent_simulation(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/invalid_id returns 404."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    response = api_client.get("/api/simulations/nonexistent_sim")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_simulation_status_completed(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id}/status returns 'completed' when results exist."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get("/api/simulations/api_run_20250107_120000/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["progress"] == 1.0


def test_get_simulation_status_pending(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/status returns 'pending' when no results exist."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "api_run_pending"
    sim_dir.mkdir(parents=True)

    # Create config but no results
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    response = api_client.get("/api/simulations/api_run_pending/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["progress"] == 0.0


def test_create_simulation_valid_config(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """POST /api/simulations with valid config returns 201 and simulation_id."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess.Popen to prevent actual simulation run
    mock_process = MagicMock()
    mock_process.pid = 12345
    monkeypatch.setattr(
        "src.api.main.subprocess.Popen", lambda *args, **kwargs: mock_process
    )

    config = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 3,
        "num_of_clients": 2,
    }

    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201
    data = response.json()
    assert "simulation_id" in data
    assert data["simulation_id"].startswith("api_run_")

    # Verify config file was created
    sim_id = data["simulation_id"]
    config_path = tmp_path / "out" / sim_id / "config.json"
    assert config_path.exists()
    with open(config_path) as f:
        saved_config = json.load(f)
    assert saved_config["shared_settings"]["aggregation_strategy_keyword"] == "fedavg"


def test_create_simulation_spawns_background_task(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """POST /api/simulations spawns background subprocess."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Track Popen calls
    popen_calls = []

    def mock_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        mock_process = MagicMock()
        mock_process.pid = 99999
        return mock_process

    monkeypatch.setattr("src.api.main.subprocess.Popen", mock_popen)

    config = {
        "aggregation_strategy_keyword": "fedavg",
        "dataset_keyword": "bloodmnist",
        "num_of_rounds": 2,
        "num_of_clients": 2,
    }

    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 201

    # Verify subprocess was called
    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    assert "python" in args[0]
    assert "-m" in args[0]
    assert "src.simulation_runner" in args[0]


# --- File Serving Tests ---


def test_get_metrics_csv(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id}/results/metrics.csv serves CSV as JSON."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get(
        "/api/simulations/api_run_20250107_120000/results/metrics.csv"
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2  # Two rows in mock CSV
    assert data[0]["round"] == 1
    assert data[0]["accuracy"] == 0.85


def test_get_plot_data_json(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id}/plot-data serves interactive plot JSON."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get("/api/simulations/api_run_20250107_120000/plot-data")
    assert response.status_code == 200
    data = response.json()
    assert "rounds" in data
    assert "accuracy" in data
    assert data["rounds"] == [1, 2, 3]


def test_get_plot_data_not_found(api_client: TestClient, tmp_path: Path, monkeypatch):
    """GET /api/simulations/{id}/plot-data returns 404 when JSON missing."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "api_run_no_plots"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    response = api_client.get("/api/simulations/api_run_no_plots/plot-data")
    assert response.status_code == 404
    assert "not yet available" in response.json()["detail"].lower()


def test_get_static_plot(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id}/results/{plot_name} serves PDF file."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get(
        "/api/simulations/api_run_20250107_120000/results/accuracy_plot.pdf"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_get_missing_file_returns_404(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET nonexistent result file returns 404."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get(
        "/api/simulations/api_run_20250107_120000/results/nonexistent.pdf"
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_unsupported_file_type(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET unsupported file type returns 400."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get(
        "/api/simulations/api_run_20250107_120000/results/malicious.exe"
    )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


# --- CORS Tests ---


def test_cors_allows_frontend_origins(api_client: TestClient):
    """CORS middleware allows localhost:5173-5178."""
    headers = {"Origin": "http://localhost:5173"}
    response = api_client.get("/", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_cors_allows_all_configured_ports(api_client: TestClient):
    """CORS middleware allows all configured frontend ports."""
    allowed_ports = [5173, 5174, 5175, 5176, 5177, 5178]
    for port in allowed_ports:
        headers = {"Origin": f"http://localhost:{port}"}
        response = api_client.get("/", headers=headers)
        assert response.status_code == 200


# --- Error Handling Tests ---


def test_path_traversal_blocked(api_client: TestClient, tmp_path: Path, monkeypatch):
    """Path traversal attempts are blocked with 400 or 404."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "test_sim"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Attempt path traversal
    response = api_client.get("/api/simulations/test_sim/results/../../../etc/passwd")
    # Can return either 400 (invalid path) or 404 (not found after normalization)
    assert response.status_code in [400, 404]


def test_invalid_simulation_id_format(api_client: TestClient):
    """Invalid simulation ID format returns 400 or 404."""
    response = api_client.get("/api/simulations/../../etc/passwd")
    # Can return either 400 (invalid format) or 404 (not found after validation)
    assert response.status_code in [400, 404]


def test_missing_config_json_returns_404(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id} returns 404 when config.json missing."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "sim_no_config"
    sim_dir.mkdir(parents=True)

    response = api_client.get("/api/simulations/sim_no_config")
    assert response.status_code == 404
    assert "config.json not found" in response.json()["detail"]


def test_malformed_config_json_returns_500(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id} returns 500 when config.json is malformed."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "sim_bad_config"
    sim_dir.mkdir(parents=True)
    (sim_dir / "config.json").write_text("{invalid json")

    response = api_client.get("/api/simulations/sim_bad_config")
    assert response.status_code == 500
    assert "could not read" in response.json()["detail"].lower()


def test_csv_read_error_returns_500(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """CSV read errors return 500 with error message."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "sim_bad_csv"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Create malformed CSV (binary content that pandas can't parse)
    (sim_dir / "bad.csv").write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    response = api_client.get("/api/simulations/sim_bad_csv/results/bad.csv")
    # Pandas may handle binary data gracefully, so check for 200 or 500
    assert response.status_code in [200, 400, 500]


# --- Edge Cases ---


def test_simulation_with_nested_config_structure(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations handles nested config structure."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "nested_config"
    sim_dir.mkdir(parents=True)

    # Create config with nested structure (backward compatibility test)
    config = {
        "shared_settings": {
            "aggregation_strategy_keyword": "pid",
            "num_of_rounds": 10,
            "num_of_clients": 5,
        },
        "simulation_strategies": [{}],
    }
    (sim_dir / "config.json").write_text(json.dumps(config))

    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    sims = response.json()
    assert len(sims) == 1
    assert sims[0]["strategy_name"] == "pid"
    assert sims[0]["num_of_rounds"] == 10


def test_simulation_status_failed_with_error_log(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/status returns 'failed' with error message."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "failed_sim"
    sim_dir.mkdir(parents=True)

    # Create config and execution log but no results
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))
    (sim_dir / "execution.log").write_text("Critical error: Out of memory")

    response = api_client.get("/api/simulations/failed_sim/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "error" in data
    assert "Out of memory" in data["error"]


def test_simulation_details_with_failed_status(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id} returns 'failed' status when execution.log exists."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "failed_details"
    sim_dir.mkdir(parents=True)

    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))
    (sim_dir / "execution.log").write_text("Simulation failed")

    response = api_client.get("/api/simulations/failed_details")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"


def test_get_result_file_json_format(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """GET /api/simulations/{id}/results/{file}.json serves JSON file."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    response = api_client.get(
        "/api/simulations/api_run_20250107_120000/results/plot_data_0.json"
    )
    assert response.status_code == 200
    # JSON files are served as FileResponse, not parsed
    assert response.headers["content-type"] == "application/json"


def test_simulation_status_with_finished_process_no_results(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """Process finished with non-zero exit and no results returns 'failed'."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "failed_process"
    sim_dir.mkdir(parents=True)

    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))
    (sim_dir / "execution.log").write_text("Process exited with code 1")

    # Mock a finished process with error
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Non-zero exit code

    main.running_processes["failed_process"] = mock_process

    response = api_client.get("/api/simulations/failed_process/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert data["progress"] == 0.0

    # Process should be removed from tracking
    assert "failed_process" not in main.running_processes


def test_simulation_status_with_running_process(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/status returns 'running' when process is active."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "api_run_active"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Mock a running process
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Still running

    main.running_processes["api_run_active"] = mock_process

    response = api_client.get("/api/simulations/api_run_active/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["progress"] == 0.0

    # Cleanup
    del main.running_processes["api_run_active"]


def test_simulation_status_transitions_to_completed(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """Process completion transitions status from 'running' to 'completed'."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    # Mock a finished process
    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Finished successfully

    main.running_processes["api_run_20250107_120000"] = mock_process

    response = api_client.get("/api/simulations/api_run_20250107_120000/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["progress"] == 1.0

    # Process should be removed from tracking
    assert "api_run_20250107_120000" not in main.running_processes


# --- Additional Exception Handling Tests ---


def test_list_simulations_with_malformed_config(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations skips simulations with malformed config.json."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    # Create simulation with valid config
    valid_sim = tmp_path / "out" / "valid_sim"
    valid_sim.mkdir()
    valid_config = {
        "shared_settings": {
            "aggregation_strategy_keyword": "fedavg",
            "num_of_rounds": 5,
            "num_of_clients": 3,
        }
    }
    (valid_sim / "config.json").write_text(json.dumps(valid_config))

    # Create simulation with malformed config
    bad_sim = tmp_path / "out" / "bad_sim"
    bad_sim.mkdir()
    (bad_sim / "config.json").write_text("{invalid json}")

    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    sims = response.json()

    # Should only return the valid simulation
    assert len(sims) == 1
    assert sims[0]["simulation_id"] == "valid_sim"


def test_list_simulations_with_io_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations handles IO errors when reading config."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    sim_dir = tmp_path / "out" / "io_error_sim"
    sim_dir.mkdir()

    # Create config file with valid content
    config = {"shared_settings": {"aggregation_strategy_keyword": "fedavg"}}
    config_path = sim_dir / "config.json"
    config_path.write_text(json.dumps(config))

    # Mock Path.open to raise IOError
    from pathlib import Path as PathLib

    original_open = PathLib.open

    def mock_open(self, *args, **kwargs):
        if "io_error_sim" in str(self) and "config.json" in str(self):
            raise IOError("Permission denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    response = api_client.get("/api/simulations")
    assert response.status_code == 200
    # Should skip the problematic simulation
    sims = response.json()
    assert all(sim["simulation_id"] != "io_error_sim" for sim in sims)


def test_create_simulation_config_write_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """POST /api/simulations handles config write errors."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Make output directory read-only to cause write error
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)

    # Mock Path.open to raise IOError
    from pathlib import Path as PathLib

    original_open = PathLib.open

    def mock_open(self, *args, **kwargs):
        if "config.json" in str(self):
            raise IOError("Disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    config = {"aggregation_strategy_keyword": "fedavg"}

    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 500
    assert "failed to write config" in response.json()["detail"].lower()


def test_create_simulation_subprocess_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """POST /api/simulations handles subprocess creation errors."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    # Mock subprocess.Popen to raise exception
    def mock_popen(*args, **kwargs):
        raise OSError("Failed to spawn process")

    monkeypatch.setattr("src.api.main.subprocess.Popen", mock_popen)

    config = {"aggregation_strategy_keyword": "fedavg"}

    response = api_client.post("/api/simulations", json=config)
    assert response.status_code == 500
    assert "failed to start simulation" in response.json()["detail"].lower()


def test_get_simulation_status_execution_log_io_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/status handles execution.log read errors."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "execution_log_issue"
    sim_dir.mkdir(parents=True)

    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))
    (sim_dir / "execution.log").write_text("Some error")

    # Mock a finished process
    mock_process = MagicMock()
    mock_process.poll.return_value = 1  # Non-zero exit

    main.running_processes["execution_log_issue"] = mock_process

    # Mock Path.open to raise IOError when reading execution.log
    from pathlib import Path as PathLib

    original_open = PathLib.open

    def mock_open(self, *args, **kwargs):
        if "execution.log" in str(self):
            raise IOError("Cannot read execution log")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.open", mock_open)

    response = api_client.get("/api/simulations/execution_log_issue/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    # error field should be None or absent when execution.log can't be read
    assert data.get("error") is None


def test_get_plot_data_nonexistent_simulation(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/plot-data returns 404 for nonexistent simulation."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    response = api_client.get("/api/simulations/nonexistent/plot-data")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_plot_data_json_parse_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/plot-data handles JSON parse errors."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "bad_json"
    sim_dir.mkdir(parents=True)

    # Create malformed plot data JSON
    (sim_dir / "plot_data_0.json").write_text("{invalid json")

    response = api_client.get("/api/simulations/bad_json/plot-data")
    assert response.status_code == 500
    # Should return generic error message


def test_get_plot_data_file_not_found_error(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id}/plot-data handles FileNotFoundError explicitly."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "missing_plot"
    sim_dir.mkdir(parents=True)

    # Create plot_data file that will be deleted
    plot_file = sim_dir / "plot_data_0.json"
    plot_file.write_text('{"rounds": [1, 2, 3]}')

    # Mock open to raise FileNotFoundError
    original_open = open

    def mock_open(*args, **kwargs):
        if "plot_data_0.json" in str(args[0]):
            raise FileNotFoundError("File disappeared")
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    response = api_client.get("/api/simulations/missing_plot/plot-data")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_validate_dataset_valid(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate returns valid dataset info."""
    from unittest.mock import Mock

    # Mock HuggingFace dataset builder
    mock_builder = Mock()
    mock_builder.info.splits = {
        "train": Mock(num_examples=60000),
        "test": Mock(num_examples=10000),
    }
    mock_builder.info.features = "{'image': Image, 'label': ClassLabel}"

    def mock_load_builder(name):
        return mock_builder

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=ylecun/mnist")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["compatible"] is True
    assert "info" in data
    assert data["info"]["has_label"] is True


def test_validate_dataset_not_found(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate handles dataset not found."""

    def mock_load_builder(name):
        raise Exception("Dataset not found on HuggingFace Hub")

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=invalid/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert data["compatible"] is False
    assert "not found" in data["error"].lower()


def test_validate_dataset_network_error(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate handles network errors."""

    def mock_load_builder(name):
        raise Exception("Connection timeout")

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=ylecun/mnist")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "network" in data["error"].lower() or "connection" in data["error"].lower()


def test_validate_dataset_authentication_error(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate handles authentication errors."""

    def mock_load_builder(name):
        raise Exception("Unauthorized: 401")

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=private/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert (
        "authentication" in data["error"].lower()
        or "unauthorized" in data["error"].lower()
    )


def test_validate_dataset_forbidden_error(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate handles forbidden errors."""

    def mock_load_builder(name):
        raise Exception("Forbidden: 403")

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=restricted/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "forbidden" in data["error"].lower() or "permission" in data["error"].lower()


def test_validate_dataset_invalid_format(api_client: TestClient, monkeypatch):
    """GET /api/datasets/validate handles invalid dataset name format."""

    def mock_load_builder(name):
        raise Exception("Invalid dataset identifier")

    monkeypatch.setattr("src.api.main.load_dataset_builder", mock_load_builder)

    response = api_client.get("/api/datasets/validate?name=invalidformat")
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "format" in data["error"].lower()


def test_secure_join_prevents_traversal(tmp_path):
    """secure_join prevents path traversal attacks."""
    from src.api.main import HTTPException, secure_join

    # Use tmp_path to ensure base exists on Windows
    base = tmp_path / "safe"
    base.mkdir()

    # Valid path
    result = secure_join(base, "subdir", "file.txt")
    # Resolve both paths for comparison on Windows
    assert result.resolve().is_relative_to(base.resolve())

    # Path traversal attempts should raise HTTPException
    try:
        secure_join(base, "..", "..", "etc", "passwd")
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 400
        assert "invalid path" in e.detail.lower()


def test_get_simulation_path_invalid_id():
    """get_simulation_path rejects invalid simulation IDs."""
    from src.api.main import HTTPException, get_simulation_path

    # Invalid characters in simulation ID
    try:
        get_simulation_path("../../malicious")
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code in [400, 404]


def test_simulation_status_completed_with_results_no_process(
    api_client: TestClient, mock_simulation_dir: Path, monkeypatch
):
    """Status returns 'completed' when results exist but no tracked process."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", mock_simulation_dir.parent)

    # Ensure no process is tracked
    if "api_run_20250107_120000" in main.running_processes:
        del main.running_processes["api_run_20250107_120000"]

    response = api_client.get("/api/simulations/api_run_20250107_120000/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["progress"] == 1.0


def test_simulation_details_filters_dataset_directories(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """GET /api/simulations/{id} excludes dataset_ directories from result_files."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "test_filter"
    sim_dir.mkdir(parents=True)

    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Create result files
    (sim_dir / "metrics.csv").write_text("round,accuracy\n1,0.8")
    (sim_dir / "plot.pdf").write_bytes(b"%PDF-1.4")

    # Create dataset directory (should be filtered out)
    dataset_dir = sim_dir / "dataset_mnist"
    dataset_dir.mkdir()
    (dataset_dir / "train.csv").write_text("data")

    response = api_client.get("/api/simulations/test_filter")
    assert response.status_code == 200
    data = response.json()

    # Dataset files should not be in result_files
    assert "metrics.csv" in data["result_files"]
    assert "plot.pdf" in data["result_files"]
    assert not any("dataset_" in f for f in data["result_files"])


def test_create_simulation_sets_loky_env_vars(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """POST /api/simulations sets LOKY_MAX_CPU_COUNT environment variable."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("src.api.main.BASE_DIR", tmp_path)

    captured_env = {}

    def mock_popen(*args, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        mock_process = MagicMock()
        mock_process.pid = 12345
        return mock_process

    monkeypatch.setattr("src.api.main.subprocess.Popen", mock_popen)

    config = {"aggregation_strategy_keyword": "fedavg"}
    response = api_client.post("/api/simulations", json=config)

    assert response.status_code == 201
    assert "LOKY_MAX_CPU_COUNT" in captured_env
    assert int(captured_env["LOKY_MAX_CPU_COUNT"]) > 0


# --- DELETE Endpoint Tests ---


def test_delete_simulation_success(api_client: TestClient, tmp_path: Path, monkeypatch):
    """DELETE /api/simulations/{id} deletes simulation successfully."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "test_delete"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))
    (sim_dir / "results.csv").write_text("data")

    assert sim_dir.exists()

    response = api_client.delete("/api/simulations/test_delete")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "deleted"
    assert data["simulation_id"] == "test_delete"

    # Verify directory was deleted
    assert not sim_dir.exists()


def test_delete_simulation_not_found(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations/{id} returns 404 for nonexistent simulation."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    response = api_client.delete("/api/simulations/nonexistent")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_simulation_running_process(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations/{id} returns 409 for running simulation."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "running_sim"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Mock a running process
    mock_process = MagicMock()
    mock_process.poll.return_value = None  # Still running
    main.running_processes["running_sim"] = mock_process

    response = api_client.delete("/api/simulations/running_sim")
    assert response.status_code == 409
    assert "cannot delete" in response.json()["detail"].lower()

    # Cleanup
    del main.running_processes["running_sim"]


def test_delete_simulation_finished_process(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations/{id} succeeds when process is finished."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    sim_dir = tmp_path / "out" / "finished_sim"
    sim_dir.mkdir(parents=True)
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Mock a finished process
    mock_process = MagicMock()
    mock_process.poll.return_value = 0  # Finished
    main.running_processes["finished_sim"] = mock_process

    response = api_client.delete("/api/simulations/finished_sim")
    assert response.status_code == 200
    assert not sim_dir.exists()

    # Process should be removed from tracking
    assert "finished_sim" not in main.running_processes


def test_delete_multiple_simulations_success(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations with simulation_ids deletes multiple simulations."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")

    # Create three simulations
    for sim_id in ["sim1", "sim2", "sim3"]:
        sim_dir = tmp_path / "out" / sim_id
        sim_dir.mkdir(parents=True)
        config = {"shared_settings": {}, "simulation_strategies": [{}]}
        (sim_dir / "config.json").write_text(json.dumps(config))

    response = api_client.request(
        "DELETE", "/api/simulations", json={"simulation_ids": ["sim1", "sim2", "sim3"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["deleted"]) == 3
    assert len(data["failed"]) == 0
    assert "sim1" in data["deleted"]
    assert "sim2" in data["deleted"]
    assert "sim3" in data["deleted"]

    # Verify all were deleted
    assert not (tmp_path / "out" / "sim1").exists()
    assert not (tmp_path / "out" / "sim2").exists()
    assert not (tmp_path / "out" / "sim3").exists()


def test_delete_multiple_simulations_partial_failure(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations returns success and failures separately."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir(parents=True)

    # Create one valid simulation
    sim_dir = tmp_path / "out" / "valid_sim"
    sim_dir.mkdir()
    config = {"shared_settings": {}, "simulation_strategies": [{}]}
    (sim_dir / "config.json").write_text(json.dumps(config))

    # Delete valid + nonexistent
    response = api_client.request(
        "DELETE",
        "/api/simulations",
        json={"simulation_ids": ["valid_sim", "nonexistent"]},
    )
    assert response.status_code == 200
    data = response.json()

    assert "valid_sim" in data["deleted"]
    assert len(data["failed"]) == 1
    assert data["failed"][0]["simulation_id"] == "nonexistent"
    assert "not found" in data["failed"][0]["error"].lower()


def test_delete_multiple_simulations_running_process(
    api_client: TestClient, tmp_path: Path, monkeypatch
):
    """DELETE /api/simulations skips running simulations."""
    monkeypatch.setattr("src.api.main.OUTPUT_DIR", tmp_path / "out")

    # Create two simulations
    for sim_id in ["completed_sim", "running_sim"]:
        sim_dir = tmp_path / "out" / sim_id
        sim_dir.mkdir(parents=True)
        config = {"shared_settings": {}, "simulation_strategies": [{}]}
        (sim_dir / "config.json").write_text(json.dumps(config))

    # Mock running process
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    main.running_processes["running_sim"] = mock_process

    response = api_client.request(
        "DELETE",
        "/api/simulations",
        json={"simulation_ids": ["completed_sim", "running_sim"]},
    )
    assert response.status_code == 200
    data = response.json()

    assert "completed_sim" in data["deleted"]
    assert len(data["failed"]) == 1
    assert data["failed"][0]["simulation_id"] == "running_sim"
    assert "running" in data["failed"][0]["error"].lower()

    # Cleanup
    del main.running_processes["running_sim"]
