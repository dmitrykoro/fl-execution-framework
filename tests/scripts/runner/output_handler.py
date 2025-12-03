"""Real-time output handling for subprocesses."""

import subprocess
from typing import List, Optional, Callable
from pathlib import Path


class OutputStreamHandler:
    """Handle real-time subprocess output streaming with device detection."""

    def __init__(self, capture_for_parsing: bool = True):
        """Initialize output handler.

        Args:
            capture_for_parsing: Whether to buffer output for parsing (default: True)
        """
        self.capture_for_parsing = capture_for_parsing
        self.output_buffer: List[str] = []

    def detect_device_from_output(
        self, output_lines: Optional[List[str]] = None
    ) -> Optional[str]:
        """Parse output to detect actual device used.

        Args:
            output_lines: Lines to parse (uses buffer if not provided)

        Returns:
            "cuda" if GPU was used, "cpu" if CPU was used, None if couldn't detect
        """
        lines = output_lines if output_lines is not None else self.output_buffer

        for line in lines:
            # Check for CUDA/GPU usage
            if "Using CUDA" in line or "CUDA GPU" in line:
                return "cuda"
            # Check for CPU usage
            if "Using CPU" in line or "Using device: CPU" in line:
                return "cpu"
        return None

    def stream_output(
        self,
        process: subprocess.Popen,
        line_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Stream output from process line-by-line in real-time.

        Args:
            process: Popen process instance
            line_callback: Optional callback for each line (default: print to stdout)
        """
        if process.stdout is None:
            return

        # Default callback: print to stdout
        def default_callback(line: str) -> None:
            print(line, end="")

        if line_callback is None:
            line_callback = default_callback

        # Stream line by line
        for line in process.stdout:
            line_callback(line)

            # Buffer for parsing if enabled
            if self.capture_for_parsing:
                self.output_buffer.append(line)

    def clear_buffer(self) -> None:
        """Clear the output buffer."""
        self.output_buffer.clear()

    def get_buffer(self) -> List[str]:
        """Get copy of output buffer.

        Returns:
            List of output lines
        """
        return self.output_buffer.copy()


def create_subprocess(
    cmd: List[str], cwd: Path, env: dict, encoding: str = "utf-8"
) -> subprocess.Popen:
    """Create subprocess with optimal settings for real-time output.

    Args:
        cmd: Command and arguments list
        cwd: Working directory
        env: Environment variables
        encoding: Output encoding (default: utf-8)

    Returns:
        Popen process instance
    """
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        encoding=encoding,  # Python 3.6+ feature for automatic text mode
        bufsize=1,  # Line buffered for real-time output
        env=env,
    )
