# src/api/main.py

import json
import multiprocessing
import re
import shutil
import subprocess
import datetime
import logging
import os
from pathlib import Path
import traceback
from typing import Optional, Dict, Any, List, Union
from datasets import load_dataset_builder
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import psutil
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("=== API main.py loaded with latin-1 CSV encoding fix ===")

app = FastAPI()

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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "out"

running_processes: Dict[str, subprocess.Popen] = {}

# --- Pydantic Models ---


class SimulationConfig(BaseModel):
    display_name: Optional[str] = None
    aggregation_strategy_keyword: Optional[str] = None
    remove_clients: Optional[str] = None
    begin_removing_from_round: Optional[int] = None
    dataset_keyword: Optional[str] = None
    dataset_source: Optional[str] = None  # 'local' or 'huggingface'
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
    hf_dataset_name: Optional[str] = None
    partitioning_strategy: Optional[str] = None
    partitioning_params: Optional[dict] = None
    transformer_model: Optional[str] = None
    max_seq_length: Optional[int] = None
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    use_lora: Optional[bool] = None
    lora_rank: Optional[int] = None


class SimulationMetadata(BaseModel):
    simulation_id: str
    display_name: Optional[str] = None
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
        final_path = (base / Path(*paths)).resolve()
        final_path.relative_to(base.resolve())
        return final_path
    except (ValueError, FileNotFoundError):
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
    """Scans the output directory for all simulation runs and returns their metadata."""
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

                    settings = config.get("shared_settings", config)

                    created_at = datetime.datetime.fromtimestamp(
                        sim_dir.stat().st_ctime
                    ).isoformat()

                    simulations.append(
                        SimulationMetadata(
                            simulation_id=sim_dir.name,
                            display_name=settings.get("display_name"),
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
    """Returns the configuration and a list of result files for a specific simulation."""
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
            rel_path = item.relative_to(sim_path)
            rel_path_str = str(rel_path).replace("\\", "/")
            if not rel_path_str.startswith("dataset_"):
                result_files.append(rel_path_str)

    # Check if simulation was manually stopped (must be checked first)
    stopped_marker = sim_path / ".stopped"
    if stopped_marker.is_file():
        status = "stopped"
    elif simulation_id in running_processes:
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
    result_filename: str,
    sim_path: Path = Depends(get_simulation_path),
    download: bool = False,
) -> Union[FileResponse, JSONResponse]:
    """
    Serves a specific result file (e.g., a plot image, PDF, or CSV).

    Args:
        result_filename: Name of the result file to retrieve
        sim_path: Path to simulation directory (injected by dependency)
        download: If True, return file for download instead of JSON (CSV only)
    """
    if not result_filename.endswith((".png", ".pdf", ".csv", ".json")):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_path = secure_join(sim_path, result_filename)

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")

    if result_filename.endswith(".csv"):
        if download:
            filename = Path(result_filename).name
            return FileResponse(
                file_path,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        try:
            df = pd.read_csv(file_path, encoding="latin-1")
            return JSONResponse(content=df.to_dict(orient="records"))
        except Exception as e:
            logger.error(f"Failed to read or parse CSV file {file_path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process CSV file: {str(e)}"
            )

    return FileResponse(file_path)


@app.post("/api/simulations", status_code=201)
async def create_simulation(config: SimulationConfig) -> Dict[str, str]:
    """Creates and runs a new simulation from a configuration payload."""
    config_dict = config.model_dump(exclude_unset=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_id = f"api_run_{timestamp}"

    output_sim_path = OUTPUT_DIR / simulation_id
    output_sim_path.mkdir(parents=True, exist_ok=True)

    config_filepath = output_sim_path / "config.json"

    wrapped_config = {
        "shared_settings": config_dict,
        "simulation_strategies": [{}],
    }

    try:
        with config_filepath.open("w") as f:
            json.dump(wrapped_config, f, indent=4)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config file: {e}")

    try:
        error_log_path = output_sim_path / "error.log"

        env = dict(os.environ)
        try:
            physical_cores = (
                psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            )
            env["LOKY_MAX_CPU_COUNT"] = str(physical_cores)
        except ImportError:
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
    """Returns the current status of a simulation (running/completed/failed/stopped)."""
    # Check if simulation was manually stopped
    stopped_marker = sim_path / ".stopped"
    stopped_exists = stopped_marker.is_file()
    # print(f"DEBUG: Checking {stopped_marker}, exists={stopped_exists}", flush=True)
    if stopped_exists:
        # print(f"DEBUG: Returning stopped status for {simulation_id}", flush=True)
        return {"status": "stopped", "progress": 0.0}

    if simulation_id in running_processes:
        process = running_processes[simulation_id]
        poll_result = process.poll()

        if poll_result is None:
            return {"status": "running", "progress": 0.0}
        else:
            del running_processes[simulation_id]

            result_files = list(sim_path.glob("*.pdf")) + list(
                sim_path.glob("csv/*.csv")
            )
            if result_files:
                return {"status": "completed", "progress": 1.0}

            if poll_result == 0:
                return {"status": "completed", "progress": 1.0}
            else:
                error_log_path = sim_path / "error.log"
                error_message = None
                if error_log_path.is_file():
                    try:
                        with error_log_path.open("r") as f:
                            error_message = f.read().strip()
                    except IOError:
                        pass
                return {"status": "failed", "progress": 0.0, "error": error_message}

    result_files = list(sim_path.glob("*.pdf")) + list(sim_path.glob("csv/*.csv"))
    if result_files:
        return {"status": "completed", "progress": 1.0}

    error_log_path = sim_path / "error.log"
    if error_log_path.is_file() and not result_files:
        try:
            with error_log_path.open("r") as f:
                error_message = f.read().strip()
            if error_message:
                return {"status": "failed", "progress": 0.0, "error": error_message}
        except IOError:
            pass

    return {"status": "pending", "progress": 0.0}


@app.delete("/api/simulations/{simulation_id}", status_code=200)
def delete_simulation(
    sim_path: Path = Depends(get_simulation_path), simulation_id: str = ""
) -> Dict[str, str]:
    """Delete a simulation and all its files."""
    if simulation_id in running_processes:
        process = running_processes[simulation_id]
        if process.poll() is None:
            raise HTTPException(
                status_code=409, detail="Cannot delete a running simulation."
            )
        del running_processes[simulation_id]

    try:
        shutil.rmtree(sim_path)
        logger.info(f"Deleted simulation: {simulation_id}")
        return {"message": "deleted", "simulation_id": simulation_id}
    except Exception as e:
        logger.error(f"Failed to delete simulation {simulation_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete simulation: {str(e)}"
        )


@app.delete("/api/simulations", status_code=200)
def delete_multiple_simulations(
    simulation_ids: List[str] = Body(..., embed=True),
) -> Dict[str, Any]:
    """Delete multiple simulations at once."""
    deleted = []
    failed = []

    for simulation_id in simulation_ids:
        try:
            if not simulation_id.isalnum() and "_" not in simulation_id:
                failed.append(
                    {"simulation_id": simulation_id, "error": "Invalid simulation ID"}
                )
                continue

            sim_path = secure_join(OUTPUT_DIR, simulation_id)

            if not sim_path.is_dir():
                failed.append(
                    {"simulation_id": simulation_id, "error": "Simulation not found"}
                )
                continue

            if simulation_id in running_processes:
                process = running_processes[simulation_id]
                if process.poll() is None:
                    failed.append(
                        {
                            "simulation_id": simulation_id,
                            "error": "Simulation is running",
                        }
                    )
                    continue
                del running_processes[simulation_id]

            shutil.rmtree(sim_path)
            deleted.append(simulation_id)
            logger.info(f"Deleted simulation: {simulation_id}")

        except Exception as e:
            logger.error(f"Failed to delete simulation {simulation_id}: {e}")
            failed.append({"simulation_id": simulation_id, "error": str(e)})

    return {"deleted": deleted, "failed": failed}


@app.patch("/api/simulations/{simulation_id}/rename", status_code=200)
def rename_simulation(
    simulation_id: str,
    display_name: str = Body(..., embed=True),
    sim_path: Path = Depends(get_simulation_path),
) -> Dict[str, str]:
    """Update the display name of a simulation."""
    if not display_name or not display_name.strip():
        raise HTTPException(
            status_code=400, detail="Display name cannot be empty or whitespace only"
        )

    display_name = display_name.strip()

    if len(display_name) > 100:
        raise HTTPException(
            status_code=400, detail="Display name must be 100 characters or less"
        )

    import re

    if not re.match(r"^[a-zA-Z0-9\s\-_]+$", display_name):
        raise HTTPException(
            status_code=400,
            detail="Display name can only contain letters, numbers, spaces, hyphens, and underscores",
        )

    config_path = sim_path / "config.json"
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail="Simulation config.json not found.")

    try:
        with config_path.open("r") as f:
            config = json.load(f)

        if "shared_settings" in config:
            config["shared_settings"]["display_name"] = display_name
        else:
            config["display_name"] = display_name

        with config_path.open("w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Renamed simulation {simulation_id} to '{display_name}'")
        return {
            "message": "renamed",
            "simulation_id": simulation_id,
            "display_name": display_name,
        }

    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to rename simulation {simulation_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update simulation config: {str(e)}"
        )


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Federated Learning Simulation Framework API"}


@app.get("/api/simulations/{simulation_id}/plot-data")
async def get_plot_data(simulation_id: str) -> Dict:
    """Return JSON plot data for interactive visualization"""
    try:
        sim_dir = OUTPUT_DIR / simulation_id

        if not sim_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Simulation directory not found for {simulation_id}",
            )

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


@app.post("/api/simulations/{simulation_id}/stop", status_code=200)
def stop_simulation(simulation_id: str) -> Dict[str, str]:
    """Stop a running simulation."""
    if simulation_id not in running_processes:
        raise HTTPException(
            status_code=404, detail="Simulation is not running or does not exist."
        )

    process = running_processes[simulation_id]

    if process.poll() is not None:
        del running_processes[simulation_id]
        raise HTTPException(status_code=409, detail="Simulation has already completed.")

    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        del running_processes[simulation_id]

        # Write stopped status marker file
        sim_path = OUTPUT_DIR / simulation_id
        stopped_marker = sim_path / ".stopped"
        try:
            with stopped_marker.open("w") as f:
                f.write(f"Simulation stopped at {datetime.datetime.now().isoformat()}")
        except IOError as e:
            logger.warning(f"Failed to write stopped marker for {simulation_id}: {e}")

        logger.info(f"Stopped simulation: {simulation_id}")
        return {"message": "stopped", "simulation_id": simulation_id}
    except Exception as e:
        logger.error(f"Failed to stop simulation {simulation_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop simulation: {str(e)}"
        )


@app.get("/api/datasets/validate")
async def validate_dataset(name: str) -> Dict[str, Any]:
    """
    Validate HuggingFace dataset exists and is Flower-compatible.

    Args:
        name: HuggingFace dataset identifier (e.g., "ylecun/mnist")

    Returns:
        {
            "valid": bool,
            "compatible": bool,
            "info": {"splits": list, "num_examples": int, "features": str,
                     "has_label": bool, "key_features": list} | null,
            "error": str | null
        }
    """
    try:
        # Only load metadata to avoid downloading full dataset
        builder = load_dataset_builder(name)

        splits = list(builder.info.splits.keys())
        num_examples = sum(s.num_examples for s in builder.info.splits.values())
        features = str(builder.info.features)

        # Check for supervised learning label fields
        label_field_indicators = [
            "label",
            "labels",
            "class",
            "target",
            "fine_label",
            "coarse_label",
        ]
        has_label = any(field in features.lower() for field in label_field_indicators)

        key_features = []
        if builder.info.features:
            try:
                feature_matches = re.findall(r"(?:['\"](\w+)['\"]|(\w+))\s*:", features)
                feature_matches = [m[0] or m[1] for m in feature_matches]
                if feature_matches:
                    key_features = list(dict.fromkeys(feature_matches))[:5]
            except Exception:
                key_features = []

        compatible = True

        return {
            "valid": True,
            "compatible": compatible,
            "info": {
                "splits": splits,
                "num_examples": num_examples,
                "features": features,
                "has_label": has_label,
                "key_features": key_features,
            },
            "error": None,
        }

    except Exception as e:
        error_message = str(e)
        error_lower = error_message.lower()

        if (
            "connection" in error_lower
            or "network" in error_lower
            or "timeout" in error_lower
        ):
            error_message = "Network error: Unable to connect to HuggingFace Hub. Please check your internet connection."
        elif (
            "not found" in error_lower
            or "doesn't exist" in error_lower
            or "404" in error_lower
        ):
            error_message = f"Dataset '{name}' not found on HuggingFace Hub. Please verify the dataset name."
        elif (
            "authentication" in error_lower
            or "unauthorized" in error_lower
            or "401" in error_lower
        ):
            error_message = "Authentication error: This dataset may require HuggingFace login or access permissions."
        elif "forbidden" in error_lower or "403" in error_lower:
            error_message = (
                "Access forbidden: You may not have permission to access this dataset."
            )
        elif len(name) < 2 or "/" not in name:
            error_message = "Invalid dataset name format. Expected format: 'username/dataset-name' (e.g., 'ylecun/mnist')."
        else:
            error_message = f"Unable to validate dataset: {error_message}"

        return {
            "valid": False,
            "compatible": False,
            "info": None,
            "error": error_message,
        }
