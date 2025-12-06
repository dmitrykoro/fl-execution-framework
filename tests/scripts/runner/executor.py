"""Experiment execution logic."""

import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .timing_db import TimingDatabase

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
    exit_code: Optional[int]
    duration: float
    output_dir: Optional[str] = None


class ExperimentExecutor:
    """Executes experiments with timeout and cleanup."""

    def __init__(
        self,
        project_root: Path,
        config_subdir: str = "examples",
        log_level: str = "INFO",
        timeout: Optional[int] = None,
        cleanup_mode: str = "none",
        timing_db: Optional[TimingDatabase] = None,
        skip_gc: bool = False,
    ):
        """Initialize executor.

        Args:
            project_root: Project root directory
            config_subdir: Subdirectory under config/simulation_strategies/ (default: examples)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            timeout: Timeout in seconds per experiment, None for unlimited
            cleanup_mode: Cleanup mode (none, basic, aggressive)
            timing_db: Optional timing database to record execution times
            skip_gc: Skip manual garbage collection (rely on subprocess cleanup)
        """
        self.project_root = Path(project_root)
        self.config_subdir = config_subdir
        self.log_level = log_level
        self.timeout = timeout
        self.cleanup_mode = cleanup_mode
        self.timing_db = timing_db
        self.skip_gc = skip_gc
        self.python_exe = sys.executable
        self.log_file_handle = None
        self.log_file_path = None

    def __enter__(self):
        """Enter context manager - return self for with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - ensure log file is closed."""
        self._close_log_file()
        return False  # Don't suppress exceptions

    def _open_log_file(self) -> None:
        """Create and open a log file for capturing all experiment output."""
        # Create logs directory if it doesn't exist
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.log_file_path = logs_dir / f"experiment_batch_{timestamp}.log"

        # Open file in append mode with UTF-8 encoding
        self.log_file_handle = open(self.log_file_path, "a", encoding="utf-8")

        # Write header
        self.log_file_handle.write("=" * 80 + "\n")
        self.log_file_handle.write(f"Experiment Batch Run - {timestamp}\n")
        self.log_file_handle.write(
            f"Config Directory: config/simulation_strategies/{self.config_subdir}\n"
        )
        self.log_file_handle.write("=" * 80 + "\n\n")
        self.log_file_handle.flush()

    def _close_log_file(self) -> None:
        """Close the log file."""
        if self.log_file_handle:
            self.log_file_handle.write("\n" + "=" * 80 + "\n")
            self.log_file_handle.write("Batch execution completed\n")
            self.log_file_handle.write("=" * 80 + "\n")
            self.log_file_handle.close()
            self.log_file_handle = None

    def _kill_ray_processes(self) -> None:
        """Kill stray Ray processes to prevent resource leaks.

        On Windows, Ray's raylet.exe can become orphaned and hold GPU memory,
        file locks, and network ports. This ensures clean slate between experiments.
        """
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
            pass  # Don't crash if cleanup fails

    def _get_config_title(self, config_name: str) -> Optional[str]:
        """Extract title from config file if available.

        Args:
            config_name: Name of config file

        Returns:
            Title string if found, None otherwise
        """
        config_path = (
            self.project_root
            / "config"
            / "simulation_strategies"
            / self.config_subdir
            / config_name
        )
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("_title")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _get_config_device(self, config_name: str) -> str:
        """Extract training device from config file.

        Args:
            config_name: Name of config file

        Returns:
            Device string ("gpu", "cpu"), defaults to "gpu"
        """
        config_path = (
            self.project_root
            / "config"
            / "simulation_strategies"
            / self.config_subdir
            / config_name
        )
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Check shared_settings first, then root level
                if "shared_settings" in config:
                    return config["shared_settings"].get("training_device", "gpu")
                return config.get("training_device", "gpu")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return "gpu"  # Default to gpu if can't read

    def _detect_device_from_output(self, stderr_lines: List[str]) -> Optional[str]:
        """Parse stderr to detect actual device used.

        Args:
            stderr_lines: List of stderr output lines from experiment

        Returns:
            "gpu" if GPU was used, "cpu" if CPU was used, None if couldn't detect
        """
        for line in stderr_lines:
            # Check for CUDA/GPU usage
            if "Using CUDA" in line or "CUDA GPU" in line:
                return "gpu"
            # Check for CPU usage
            if "Using CPU" in line or "Using device: CPU" in line:
                return "cpu"
        return None

    def _get_newest_output_dir(self) -> Optional[str]:
        """Find the newest output directory in out/.

        Returns:
            Path to newest directory, or None if not found
        """
        out_dir = self.project_root / "out"
        if not out_dir.exists():
            return None

        # Get all directories with timestamp pattern
        dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"]
        if not dirs:
            return None

        # Return the newest directory by modification time
        newest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
        return str(newest_dir.relative_to(self.project_root))

    def run_experiment(
        self, config_name: str, config_index: int, total_configs: int
    ) -> ExperimentResult:
        """Run a single experiment.

        Args:
            config_name: Name of config file
            config_index: Current config index (1-based)
            total_configs: Total number of configs

        Returns:
            ExperimentResult with execution details
        """
        start_time = time.time()
        output_dir = None

        # Get human-readable title if available
        title = self._get_config_title(config_name)
        if title:
            console.print(
                f'\n[cyan][{config_index}/{total_configs}] Running "{title}"[/cyan]'
            )
            console.print(f"[dim]  ({config_name})[/dim]")
        else:
            console.print(
                f"\n[cyan][{config_index}/{total_configs}] Running {config_name}...[/cyan]"
            )

        # Write experiment header to log file
        if self.log_file_handle:
            self.log_file_handle.write("\n" + "=" * 80 + "\n")
            self.log_file_handle.write(
                f"[{config_index}/{total_configs}] Experiment: {config_name}\n"
            )
            if title:
                self.log_file_handle.write(f"Title: {title}\n")
            self.log_file_handle.write(
                f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.log_file_handle.write("=" * 80 + "\n\n")
            self.log_file_handle.flush()

        # Build command
        cmd = [
            self.python_exe,
            "src/simulation_runner.py",
            f"{self.config_subdir}/{config_name}",
            "--log-level",
            self.log_level,
        ]

        # Run with timeout and capture output for device detection
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)

        stderr_buffer = []  # Capture stderr for device detection

        try:
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout for simplicity
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )

            # Stream output line by line
            if process.stdout:
                for line in process.stdout:
                    print(line, end="")  # Print to console in real-time
                    stderr_buffer.append(line)  # Save for parsing
                    # Write to log file if available
                    if self.log_file_handle:
                        self.log_file_handle.write(line)
                        self.log_file_handle.flush()  # Ensure immediate write

            # Wait for process to complete with timeout
            try:
                returncode = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                duration = time.time() - start_time
                # self.timeout is guaranteed to be set if TimeoutExpired is raised
                assert self.timeout is not None
                timeout_str = (
                    f"{self.timeout // 60} minutes"
                    if self.timeout >= 60
                    else f"{self.timeout} seconds"
                )
                msg = f"[TIMEOUT] Timed out on {config_name} (exceeded {timeout_str})"
                console.print(f"[yellow]{msg}[/yellow]")
                if self.log_file_handle:
                    self.log_file_handle.write(f"\n{msg}\n")
                    self.log_file_handle.write(f"Duration: {duration:.1f}s\n")
                    self.log_file_handle.flush()
                output_dir = self._get_newest_output_dir()
                return ExperimentResult(
                    config_name, ExecutionResult.TIMEOUT, 124, duration, output_dir
                )

            duration = time.time() - start_time

            if returncode == 0:
                msg = f"[OK] Completed {config_index}/{total_configs} ({duration:.1f}s)"
                console.print(f"[green]{msg}[/green]")
                if self.log_file_handle:
                    self.log_file_handle.write(f"\n{msg}\n")
                    self.log_file_handle.flush()
                # Capture output directory
                output_dir = self._get_newest_output_dir()
                # Detect actual device used from output
                if self.timing_db:
                    detected_device = self._detect_device_from_output(stderr_buffer)
                    if detected_device:
                        device = detected_device
                    else:
                        # Fallback to config value if couldn't detect
                        device = self._get_config_device(config_name)
                    # Record with directory prefix (e.g., "examples/test.json")
                    config_path = f"{self.config_subdir}/{config_name}"
                    self.timing_db.record(config_path, duration, device)
                return ExperimentResult(
                    config_name,
                    ExecutionResult.SUCCESS,
                    returncode,
                    duration,
                    output_dir,
                )
            else:
                msg = f"[FAIL] Failed on {config_name} (exit code: {returncode}, {duration:.1f}s)"
                console.print(f"[red]{msg}[/red]")
                if self.log_file_handle:
                    self.log_file_handle.write(f"\n{msg}\n")
                    self.log_file_handle.flush()
                return ExperimentResult(
                    config_name,
                    ExecutionResult.FAILED,
                    returncode,
                    duration,
                    output_dir,
                )

        except Exception as e:
            duration = time.time() - start_time
            msg = f"[ERROR] Exception on {config_name}: {e}"
            console.print(f"[red]{msg}[/red]")
            if self.log_file_handle:
                self.log_file_handle.write(f"\n{msg}\n")
                self.log_file_handle.flush()
            output_dir = self._get_newest_output_dir()
            return ExperimentResult(
                config_name, ExecutionResult.FAILED, -1, duration, output_dir
            )

        finally:
            # Always kill stray Ray processes to prevent resource leaks
            self._kill_ray_processes()
            gc.collect()

    def cleanup(self, is_last: bool = False) -> None:
        """Perform cleanup between experiments.

        Args:
            is_last: Whether this is the last experiment (skip cleanup)
        """
        if is_last or self.cleanup_mode == "none":
            return

        console.print(f"[dim]Running cleanup (mode: {self.cleanup_mode})...[/dim]")

        if self.cleanup_mode in ["basic", "aggressive"]:
            # Force garbage collection (unless skip_gc is enabled)
            if not self.skip_gc:
                gc.collect()
                console.print("[dim]Garbage collection complete[/dim]")
            else:
                console.print(
                    "[dim]Skipping gc.collect() - relying on subprocess cleanup[/dim]"
                )

        if self.cleanup_mode == "aggressive":
            # Clear __pycache__ directories
            cache_dirs = list(self.project_root.rglob("__pycache__"))
            for cache_dir in cache_dirs:
                try:
                    for item in cache_dir.iterdir():
                        item.unlink()
                    cache_dir.rmdir()
                except Exception:
                    pass  # Ignore errors

            # Brief pause for resource reclaim
            console.print("[dim]Pausing 3 seconds for resource cleanup...[/dim]")
            time.sleep(3)

    def run_batch(
        self, configs: List[str], on_complete=None, on_failed=None, on_timeout=None
    ) -> dict:
        """Run a batch of experiments.

        Args:
            configs: List of config filenames
            on_complete: Callback for successful completion
            on_failed: Callback for failures
            on_timeout: Callback for timeouts

        Returns:
            Dictionary with results summary (includes "log_file_path" key)
        """
        total = len(configs)
        completed = []
        failed = []
        timedout = []
        output_dirs = {}  # Map config_name -> output_dir

        # Open log file for capturing all output
        self._open_log_file()
        console.print(f"[dim]Logging all output to: {self.log_file_path}[/dim]\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Running experiments...", total=total)

                for idx, config in enumerate(configs, start=1):
                    result = self.run_experiment(config, idx, total)

                    # Track output directory
                    if result.output_dir:
                        output_dirs[config] = result.output_dir

                    # Track results
                    if result.result == ExecutionResult.SUCCESS:
                        completed.append(config)
                        if on_complete:
                            on_complete(config)
                    elif result.result == ExecutionResult.TIMEOUT:
                        timedout.append(config)
                        if on_timeout:
                            on_timeout(config)
                    else:
                        failed.append(config)
                        if on_failed:
                            on_failed(config)

                    # Cleanup between experiments
                    self.cleanup(is_last=(idx == total))

                    progress.update(task, advance=1)
        finally:
            # Always close log file, even if there's an exception
            self._close_log_file()

        return {
            "completed": completed,
            "failed": failed,
            "timedout": timedout,
            "total": total,
            "output_dirs": output_dirs,
            "log_file_path": str(self.log_file_path) if self.log_file_path else None,
        }


def display_summary(
    results: dict,
    skipped: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
    config_subdir: Optional[str] = None,
) -> None:
    """Display summary of batch execution.

    Args:
        results: Results dictionary from run_batch
        skipped: List of skipped configs
        project_root: Optional project root for reading config titles
        config_subdir: Optional config subdirectory for reading config titles
    """
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]Execution Summary[/bold cyan]")
    console.print("=" * 50 + "\n")

    total_errors = len(results["failed"]) + len(results["timedout"])
    configs_run = results["total"]

    if skipped:
        console.print(
            f"[dim]Skipped {len(skipped)} already-completed config(s)[/dim]\n"
        )

    # Helper function to get config title
    def get_config_title(config_name: str) -> Optional[str]:
        if not project_root or not config_subdir:
            return None
        config_path = (
            project_root
            / "config"
            / "simulation_strategies"
            / config_subdir
            / config_name
        )
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("_title")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    # Get output directories mapping
    output_dirs = results.get("output_dirs", {})

    if total_errors == 0:
        if configs_run == 0:
            console.print("[green]All selected configs were already completed![/green]")
        else:
            console.print(
                f"[green]All {configs_run} config(s) completed successfully![/green]\n"
            )

            # Display completed experiments
            if results["completed"]:
                console.print("[cyan]Experiments run:[/cyan]")
                for config in results["completed"]:
                    title = get_config_title(config)
                    output_dir = output_dirs.get(config)

                    if title:
                        config_display = f"{title} [dim]({config})[/dim]"
                    else:
                        config_display = config

                    if output_dir:
                        console.print(f"  [green][OK][/green] {config_display}")
                        console.print(f"      [dim]-> {output_dir}[/dim]")
                    else:
                        console.print(f"  [green][OK][/green] {config_display}")
                console.print()
    else:
        console.print(f"[red]{total_errors}/{configs_run} config(s) failed:[/red]\n")

        # Display completed experiments (if any)
        if results["completed"]:
            console.print(f"[green]Completed ({len(results['completed'])}):[/green]")
            for config in results["completed"]:
                title = get_config_title(config)
                output_dir = output_dirs.get(config)

                if title:
                    config_display = f"{title} [dim]({config})[/dim]"
                else:
                    config_display = config

                if output_dir:
                    console.print(f"  [green][OK][/green] {config_display}")
                    console.print(f"      [dim]-> {output_dir}[/dim]")
                else:
                    console.print(f"  [green][OK][/green] {config_display}")
            console.print()

        if results["timedout"]:
            console.print(f"[yellow]Timed out ({len(results['timedout'])}):[/yellow]")
            for config in results["timedout"]:
                title = get_config_title(config)
                output_dir = output_dirs.get(config)

                if title:
                    config_display = f"{title} [dim]({config})[/dim]"
                else:
                    config_display = config

                if output_dir:
                    console.print(f"  [yellow][T][/yellow] {config_display}")
                    console.print(f"      [dim]-> {output_dir}[/dim]")
                else:
                    console.print(f"  [yellow][T][/yellow] {config_display}")
            console.print()

        if results["failed"]:
            console.print(f"[red]Failed ({len(results['failed'])}):[/red]")
            for config in results["failed"]:
                title = get_config_title(config)
                output_dir = output_dirs.get(config)

                if title:
                    config_display = f"{title} [dim]({config})[/dim]"
                else:
                    config_display = config

                if output_dir:
                    console.print(f"  [red][X][/red] {config_display}")
                    console.print(f"      [dim]-> {output_dir}[/dim]")
                else:
                    console.print(f"  [red][X][/red] {config_display}")
            console.print()

    # Display log file path if available
    log_file_path = results.get("log_file_path")
    if log_file_path:
        console.print(f"[cyan]Full output log:[/cyan] {log_file_path}\n")

    console.print("=" * 50)
