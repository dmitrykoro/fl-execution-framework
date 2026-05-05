"""Experiment execution logic."""

import gc
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rich.console import Console

console = Console()


class ExecutionResult(Enum):
    """Result of experiment execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ExperimentResult:
    """Result of running a single experiment."""
    config_name: str
    result: ExecutionResult
    exit_code: int | None
    duration: float
    output_dir: str | None = None


class ExperimentExecutor:
    """Executes experiments with timeout and cleanup."""

    def __init__(self, project_root: Path, config_subdir: str = "smoke", timeout: int | None = 3600):
        self.project_root = Path(project_root)
        self.config_subdir = config_subdir
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _get_newest_output_dir(self) -> str | None:
        """Find the newest output directory in out/."""
        out_dir = self.project_root / "out"
        if not out_dir.exists():
            return None

        dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"]
        if not dirs:
            return None

        newest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
        return str(newest_dir.relative_to(self.project_root))

    def run_experiment(self, config_name: str, config_index: int, total_configs: int) -> ExperimentResult:
        """Run a single experiment."""
        start_time = time.time()
        console.print(f"\n[cyan][{config_index}/{total_configs}] Running {config_name}...[/cyan]")

        cmd = [sys.executable, "src/simulation_runner.py", f"config/simulation_strategies/{self.config_subdir}/{config_name}"]
        
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.project_root}{os.pathsep}{existing_path}" if existing_path else str(self.project_root)

        DEBUG_OUTPUT = True  # Set False to silence simulation logs

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                timeout=self.timeout,
                capture_output=not DEBUG_OUTPUT,
                text=True,
            )
            duration = time.time() - start_time

            if result.returncode == 0:
                console.print(f"[green][OK] Completed in {duration:.1f}s[/green]")
                return ExperimentResult(
                    config_name, ExecutionResult.SUCCESS, result.returncode, duration, self._get_newest_output_dir()
                )
            else:
                console.print(f"[red][FAIL] Exit code {result.returncode} in {duration:.1f}s[/red]")
                console.print(result.stderr[-1000:] if result.stderr else "No stderr output.")
                return ExperimentResult(
                    config_name, ExecutionResult.FAILED, result.returncode, duration, self._get_newest_output_dir()
                )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            console.print(f"[yellow][TIMEOUT] Exceeded {self.timeout}s on {config_name}[/yellow]")
            return ExperimentResult(
                config_name, ExecutionResult.TIMEOUT, 124, duration, self._get_newest_output_dir()
            )
        except Exception as e:
            duration = time.time() - start_time
            console.print(f"[red][ERROR] Exception on {config_name}: {e}[/red]")
            return ExperimentResult(
                config_name, ExecutionResult.FAILED, -1, duration, self._get_newest_output_dir()
            )
        finally:
            self._kill_ray_processes()
            gc.collect()

    def _kill_ray_processes(self) -> None:
        """Kill stray Ray processes to prevent resource leaks."""
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "raylet.exe"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    ["pkill", "-9", "-f", "raylet"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def cleanup(self, is_last: bool = False) -> None:
        """Perform simple cleanup between experiments."""
        if not is_last:
            gc.collect()
            time.sleep(1)
