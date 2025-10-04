# src/api/main.py

import json
import subprocess
import datetime
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("=== API main.py loaded with latin-1 CSV encoding fix ===")

app = FastAPI()

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
        "http://localhost:5178",
        "http://127.0.0.1:5178",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration and Constants ---

# Use pathlib for robust path handling
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "out"

# In-memory storage for running processes
running_processes: Dict[str, subprocess.Popen] = {}

# --- Pydantic Models ---


class SimulationConfig(BaseModel):
    aggregation_strategy_keyword: Optional[str] = None
    remove_clients: Optional[str] = None
    begin_removing_from_round: Optional[int] = None
    dataset_keyword: Optional[str] = None
    num_of_rounds: Optional[int] = None
    num_of_clients: Optional[int] = None
    num_of_malicious_clients: Optional[int] = None
    attack_type: Optional[str] = None
    attack_ratio: Optional[float] = None
    gaussian_noise_mean: Optional[int] = None
    gaussian_noise_std: Optional[int] = None
    show_plots: Optional[str] = None
    save_plots: Optional[str] = None
    save_csv: Optional[str] = None
    training_device: Optional[str] = None
    cpus_per_client: Optional[int] = None
    gpus_per_client: Optional[float] = None
    trust_threshold: Optional[float] = None
    reputation_threshold: Optional[float] = None
    beta_value: Optional[float] = None
    num_of_clusters: Optional[int] = None
    Kp: Optional[float] = None
    Ki: Optional[float] = None
    Kd: Optional[float] = None
    num_std_dev: Optional[float] = None
    training_subset_fraction: Optional[float] = None
    min_fit_clients: Optional[int] = None
    min_evaluate_clients: Optional[int] = None
    min_available_clients: Optional[int] = None
    evaluate_metrics_aggregation_fn: Optional[str] = None
    num_of_client_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    preserve_dataset: Optional[str] = None
    num_krum_selections: Optional[int] = None
    trim_ratio: Optional[float] = None
    strict_mode: Optional[str] = None
    strategy_number: Optional[int] = None
    model_keyword: Optional[str] = None
    model_type: Optional[str] = None
    learning_rate: Optional[float] = None
    use_llm: Optional[str] = None
    llm_model: Optional[str] = None
    llm_finetuning: Optional[str] = None
    llm_task: Optional[str] = None
    llm_chunk_size: Optional[int] = None
    mlm_probability: Optional[float] = None


class SimulationMetadata(BaseModel):
    simulation_id: str
    strategy_name: str
    num_of_rounds: Union[int, str]
    num_of_clients: Union[int, str]
    created_at: Optional[str] = None


class SimulationDetails(BaseModel):
    config: Dict[str, Any]
    result_files: List[str]
    status: str


# --- Security and Path Validation ---


def secure_join(base: Path, *paths: str) -> Path:
    """Safely join a base directory with other paths, preventing path traversal."""
    try:
        # Resolve the path and check it's within the base directory
        final_path = (base / Path(*paths)).resolve()
        final_path.relative_to(base.resolve())
        return final_path
    except (ValueError, FileNotFoundError):
        # ValueError is raised by relative_to if the path is outside the base
        raise HTTPException(status_code=400, detail="Invalid path specified.")


def get_simulation_path(simulation_id: str) -> Path:
    """Dependency to validate and return a simulation path."""
    if not simulation_id.isalnum() and "_" not in simulation_id:
        raise HTTPException(status_code=400, detail="Invalid simulation ID format.")

    sim_path = secure_join(OUTPUT_DIR, simulation_id)

    if not sim_path.is_dir():
        raise HTTPException(status_code=404, detail="Simulation not found.")
    return sim_path


# --- API Endpoints ---


@app.get("/api/simulations", response_model=List[SimulationMetadata])
def get_simulations() -> List[SimulationMetadata]:
    """
    Scans the output directory for all simulation runs and returns their metadata.
    """
    simulations = []
    if not OUTPUT_DIR.is_dir():
        return []

    for sim_dir in OUTPUT_DIR.iterdir():
        if sim_dir.is_dir():
            config_path = sim_dir / "config.json"
            if config_path.is_file():
                try:
                    with config_path.open("r") as f:
                        config = json.load(f)

                    # Handle both flat and nested config structures
                    settings = config.get("shared_settings", config)

                    # Get directory creation time
                    created_at = datetime.datetime.fromtimestamp(
                        sim_dir.stat().st_ctime
                    ).isoformat()

                    simulations.append(
                        SimulationMetadata(
                            simulation_id=sim_dir.name,
                            strategy_name=settings.get(
                                "aggregation_strategy_keyword", "Unknown"
                            ),
                            num_of_rounds=settings.get("num_of_rounds", "N/A"),
                            num_of_clients=settings.get("num_of_clients", "N/A"),
                            created_at=created_at,
                        )
                    )
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(
                        f"Could not read or parse config for {sim_dir.name}: {e}"
                    )
                    continue
    return sorted(simulations, key=lambda s: s.simulation_id, reverse=True)


@app.get("/api/simulations/{simulation_id}", response_model=SimulationDetails)
def get_simulation_details(
    sim_path: Path = Depends(get_simulation_path), simulation_id: str = ""
) -> SimulationDetails:
    """
    Returns the configuration and a list of result files for a specific simulation.
    """
    config_path = sim_path / "config.json"
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail="Simulation config.json not found.")

    try:
        with config_path.open("r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        raise HTTPException(status_code=500, detail="Could not read simulation config.")

    result_files = []
    for item in sim_path.rglob("*"):
        if (
            item.is_file()
            and item.name != "config.json"
            and item.suffix in [".png", ".pdf", ".csv", ".json"]
        ):
            # Get relative path from sim_path
            rel_path = item.relative_to(sim_path)
            rel_path_str = str(rel_path).replace("\\", "/")
            # Skip dataset directories (contain raw training data, not results)
            if not rel_path_str.startswith("dataset_"):
                result_files.append(rel_path_str)

    # Determine status
    if simulation_id in running_processes:
        process = running_processes[simulation_id]
        if process.poll() is None:
            status = "running"
        else:
            del running_processes[simulation_id]
            status = (
                "completed"
                if result_files or list(sim_path.glob("*.pdf"))
                else "failed"
            )
    else:
        # Check for existing results
        has_results = result_files or list(sim_path.glob("*.pdf"))
        error_log = sim_path / "error.log"
        if has_results:
            status = "completed"
        elif error_log.is_file():
            status = "failed"
        else:
            status = "pending"

    return SimulationDetails(config=config, result_files=result_files, status=status)


@app.get(
    "/api/simulations/{simulation_id}/results/{result_filename:path}",
    response_model=None,
)
def get_result_file(
    result_filename: str, sim_path: Path = Depends(get_simulation_path)
) -> Union[FileResponse, JSONResponse]:
    """
    Serves a specific result file (e.g., a plot image, PDF, or CSV).
    """
    if not result_filename.endswith((".png", ".pdf", ".csv", ".json")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_path = secure_join(sim_path, result_filename)

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")

    if result_filename.endswith(".csv"):
        try:
            import pandas as pd

            df = pd.read_csv(file_path, encoding="latin-1")
            return JSONResponse(content=df.to_dict(orient="records"))
        except Exception as e:
            import traceback

            logger.error(f"Failed to read or parse CSV file {file_path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process CSV file: {str(e)}"
            )

    return FileResponse(file_path)


@app.post("/api/simulations", status_code=201)
async def create_simulation(config: SimulationConfig) -> Dict[str, str]:
    """
    Creates and runs a new simulation from a configuration payload.
    """
    config_dict = config.dict(exclude_unset=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_id = f"api_run_{timestamp}"

    output_sim_path = OUTPUT_DIR / simulation_id
    output_sim_path.mkdir(parents=True, exist_ok=True)

    config_filepath = output_sim_path / "config.json"

    # Wrap config in the expected structure for ConfigLoader
    wrapped_config = {
        "shared_settings": config_dict,
        "simulation_strategies": [
            {}
        ],  # Single empty strategy inherits all shared_settings
    }

    try:
        with config_filepath.open("w") as f:
            json.dump(wrapped_config, f, indent=4)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config file: {e}")

    try:
        # Run as a module to preserve package structure
        # Redirect stderr to capture error messages
        error_log_path = output_sim_path / "error.log"

        # Set up environment to suppress joblib/loky warnings
        env = dict(os.environ)
        try:
            import psutil

            physical_cores = (
                psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            )
            env["LOKY_MAX_CPU_COUNT"] = str(physical_cores)
        except ImportError:
            # If psutil not available, use cpu_count as fallback
            import multiprocessing

            env["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
        env["PYTHONWARNINGS"] = "ignore::RuntimeWarning:threadpoolctl"

        with error_log_path.open("w") as error_log:
            command = ["python", "-m", "src.simulation_runner", str(config_filepath)]
            process = subprocess.Popen(
                command, cwd=BASE_DIR, stderr=error_log, stdout=subprocess.PIPE, env=env
            )
        running_processes[simulation_id] = process
        logger.info(f"Started simulation {simulation_id} with PID {process.pid}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start simulation process: {e}"
        )

    return {"simulation_id": simulation_id}


@app.get("/api/simulations/{simulation_id}/status")
def get_simulation_status(
    sim_path: Path = Depends(get_simulation_path), simulation_id: str = ""
) -> Dict[str, Any]:
    """
    Returns the current status of a simulation (running/completed/failed).
    """
    # Check if process is still running
    if simulation_id in running_processes:
        process = running_processes[simulation_id]
        poll_result = process.poll()

        if poll_result is None:
            return {"status": "running", "progress": 0.0}
        else:
            # Process finished, remove from tracking
            del running_processes[simulation_id]

            # Check for result files first - simulation may return non-zero but still succeed
            result_files = list(sim_path.glob("*.pdf")) + list(
                sim_path.glob("csv/*.csv")
            )
            if result_files:
                return {"status": "completed", "progress": 1.0}

            # No results and exit code is 0 - completed but no output yet
            if poll_result == 0:
                return {"status": "completed", "progress": 1.0}
            else:
                # Non-zero exit and no results - truly failed
                error_log_path = sim_path / "error.log"
                error_message = None
                if error_log_path.is_file():
                    try:
                        with error_log_path.open("r") as f:
                            error_message = f.read().strip()
                    except IOError:
                        pass
                return {"status": "failed", "progress": 0.0, "error": error_message}

    # Check if results exist (simulation completed before server restart)
    result_files = list(sim_path.glob("*.pdf")) + list(sim_path.glob("csv/*.csv"))
    if result_files:
        return {"status": "completed", "progress": 1.0}

    # Check if error log exists AND no results (failed before server restart)
    # Note: error.log contains stderr which includes INFO logs, not just errors
    error_log_path = sim_path / "error.log"
    if error_log_path.is_file() and not result_files:
        try:
            with error_log_path.open("r") as f:
                error_message = f.read().strip()
            # Only treat as failed if we have content but no results
            if error_message:
                return {"status": "failed", "progress": 0.0, "error": error_message}
        except IOError:
            pass

    # Default to pending if no process and no results
    return {"status": "pending", "progress": 0.0}


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Federated Learning Simulation Framework API"}


@app.get("/api/simulations/{simulation_id}/plot-data")
async def get_plot_data(simulation_id: str) -> Dict:
    """Return JSON plot data for interactive visualization"""
    try:
        sim_dir = OUTPUT_DIR / simulation_id

        # Check if simulation directory exists
        if not sim_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Simulation directory not found for {simulation_id}",
            )

        # Find plot_data_*.json file in simulation root directory
        json_files = [
            f
            for f in os.listdir(sim_dir)
            if f.startswith("plot_data_") and f.endswith(".json")
        ]

        if not json_files:
            raise HTTPException(
                status_code=404,
                detail="Plot data not yet available - simulation may still be running",
            )

        # Use first JSON file (usually plot_data_0.json)
        json_path = sim_dir / json_files[0]

        with open(json_path, "r") as f:
            return json.load(f)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Plot data not found for simulation {simulation_id}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
