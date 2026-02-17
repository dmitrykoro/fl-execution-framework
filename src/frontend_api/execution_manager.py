from __future__ import annotations

import json
import subprocess
import datetime

from pathlib import Path
from typing import Any, Dict

from src.frontend_api.log_manager import log_path, ensure_log_dir_exists

CONFIG_DIR = Path("config/simulation_strategies/ui_generated")


def ensure_config_dir_exists() -> None:
    ensure_log_dir_exists()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def write_config_json(run_id: str, cfg: Dict[str, Any]) -> Path:
    ensure_config_dir_exists()

    full_config_path = CONFIG_DIR / f"{run_id}.json"
    full_config_path.write_text(json.dumps(cfg, indent=2))

    return full_config_path


def start_simulation_process(
        run_id: str,
        config_path: Path,
        log_level: str = "INFO"
) -> subprocess.Popen:
    """
    Spawns a new process that executes run_simulation.sh located two directories up from this file.
    stdout/stderr -> log file.
    """

    lp = log_path(run_id)
    lp.parent.mkdir(parents=True, exist_ok=True)

    # ---- determine working directory (two levels up) ----
    # this file: src/api/job_manager.py
    # two levels up -> project root
    project_root = Path(__file__).resolve().parents[2]

    script_path = project_root / "run_simulation.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"run_simulation.sh not found at {script_path}")

    # ---- command ----
    cmd = [
        str(script_path),
        str(config_path),
        str(run_id),
        "--log-level",
        log_level,
    ]

    # ---- open log file ----
    f = open(lp, "a", buffering=1)

    f.write(f"[aiohttp backend] Starting run_id={run_id}\n")
    f.write(f"[aiohttp backend] cwd={project_root}\n")
    f.write(f"[aiohttp backend] cmd={' '.join(cmd)}\n")

    # ---- start process ----
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),  # VERY IMPORTANT
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    return proc


def generate_run_id() -> str:
    return f"{str(datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S'))}_ui_invoked"
