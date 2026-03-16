from __future__ import annotations
from pathlib import Path
from typing import List

LOG_DIR = Path("logs/ui_runs")


def ensure_log_dir_exists() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_path(run_id: str) -> Path:
    ensure_log_dir_exists()
    return LOG_DIR / f"{run_id}.log"


def tail_lines(path: Path, max_lines: int = 300) -> List[str]:
    if not path.exists():
        return []

    text = path.read_text(errors="replace")
    lines = text.splitlines()

    return lines[-max_lines:]
