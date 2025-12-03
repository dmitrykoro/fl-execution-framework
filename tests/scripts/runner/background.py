"""Background process management for long-running experiments."""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List

import psutil
from rich.console import Console

console = Console()


class BackgroundRunner:
    """Manages background execution of experiments."""

    def __init__(self, state_dir: Path):
        """Initialize background runner.

        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.state_dir / "background_pid.txt"
        self.settings_file = self.state_dir / "background_settings.json"

    def start_background(
        self,
        script_path: str,
        log_level: str,
        timeout: Optional[int],
        cleanup_mode: str,
        selected_configs: List[str],
    ) -> int:
        """Start the runner in background mode.

        Args:
            script_path: Path to the main script
            log_level: Log level
            timeout: Timeout in seconds
            cleanup_mode: Cleanup mode
            selected_configs: List of configs to run

        Returns:
            Process ID of background process
        """
        # Save settings for background process
        settings = {
            "log_level": log_level,
            "timeout": timeout,
            "cleanup_mode": cleanup_mode,
            "selected_configs": selected_configs,
        }

        with open(self.settings_file, "w") as f:
            json.dump(settings, f, indent=2)

        # Build command
        cmd = [
            sys.executable,
            script_path,
            "--background-mode",
            "--settings-file",
            str(self.settings_file),
        ]

        # Get log file path
        log_file = self.state_dir / f"background_run_{os.getpid()}.log"

        # Start background process
        console.print("\n[cyan]Starting background execution...[/cyan]")

        with open(log_file, "w") as f:
            if os.name == "nt":  # Windows
                # Use CREATE_NO_WINDOW to truly detach on Windows
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    | subprocess.DETACHED_PROCESS,
                    start_new_session=True,
                )
            else:  # Unix-like
                process = subprocess.Popen(
                    cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
                )

        pid = process.pid

        # Save PID
        with open(self.pid_file, "w") as f:
            f.write(str(pid))

        # Display instructions
        self._display_background_info(pid, log_file)

        return pid

    def _display_background_info(self, pid: int, log_file: Path) -> None:
        """Display information about the background process.

        Args:
            pid: Process ID
            log_file: Path to log file
        """
        console.print("\n" + "=" * 50)
        console.print("[bold cyan]Background Process Started[/bold cyan]")
        console.print("=" * 50)
        console.print(f"[bold]PID:[/bold] {pid}")
        console.print(f"[bold]PID file:[/bold] {self.pid_file}")
        console.print(f"[bold]Log file:[/bold] {log_file}")
        console.print()
        console.print("[bold]Monitor progress:[/bold]")
        console.print(f"  tail -f {log_file}")
        console.print()
        console.print("[bold]Check if running:[/bold]")
        if os.name == "nt":
            console.print(f'  tasklist /FI "PID eq {pid}"')
        else:
            console.print(f"  ps -p {pid}")
        console.print()
        console.print("[bold]Stop background process:[/bold]")
        if os.name == "nt":
            console.print(f"  taskkill /PID {pid} /F")
        else:
            console.print(f"  kill {pid}")
        console.print()
        console.print("=" * 50)
        console.print("[green]You can now close this terminal safely.[/green]")
        console.print("=" * 50 + "\n")

    def load_settings(self) -> Optional[dict]:
        """Load settings from background settings file.

        Returns:
            Settings dictionary or None if not found
        """
        if not self.settings_file.exists():
            return None

        try:
            with open(self.settings_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def clear_settings(self) -> None:
        """Clear background settings file."""
        if self.settings_file.exists():
            self.settings_file.unlink()

    def is_running(self) -> bool:
        """Check if background process is running.

        Returns:
            True if running, False otherwise
        """
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            return psutil.pid_exists(pid)
        except (ValueError, IOError):
            return False

    def get_pid(self) -> Optional[int]:
        """Get PID of background process.

        Returns:
            PID or None if not found
        """
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None

    def stop(self) -> bool:
        """Stop the background process.

        Returns:
            True if stopped successfully, False otherwise
        """
        pid = self.get_pid()
        if not pid:
            console.print("[yellow]No background process found[/yellow]")
            return False

        try:
            process = psutil.Process(pid)
            process.terminate()
            console.print(f"[green]Background process {pid} terminated[/green]")
            self.pid_file.unlink()
            return True
        except psutil.NoSuchProcess:
            console.print("[yellow]Background process already stopped[/yellow]")
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
        except Exception as e:
            console.print(f"[red]Failed to stop background process: {e}[/red]")
            return False
