"""State management for experiment runner."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class RunnerState:
    """Persistent state for experiment runs."""

    completed_configs: List[str]
    failed_configs: List[str]
    timedout_configs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunnerState":
        """Create from dictionary."""
        return cls(
            completed_configs=data.get("completed_configs", []),
            failed_configs=data.get("failed_configs", []),
            timedout_configs=data.get("timedout_configs", []),
        )


class StateManager:
    """Manages persistent state across runs."""

    def __init__(self, state_dir: Path):
        """Initialize state manager.

        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "runner_state.json"

    def load_state(self) -> Optional[RunnerState]:
        """Load state from file.

        Returns:
            RunnerState if file exists, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                return RunnerState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load state file: {e}")
            return None

    def save_state(self, state: RunnerState) -> None:
        """Save state to file.

        Args:
            state: State to save
        """
        with open(self.state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def clear_state(self) -> None:
        """Clear state file."""
        if self.state_file.exists():
            self.state_file.unlink()

    def add_completed(self, config_name: str) -> None:
        """Add a config to completed list.

        Args:
            config_name: Name of completed config
        """
        state = self.load_state() or RunnerState([], [], [])
        if config_name not in state.completed_configs:
            state.completed_configs.append(config_name)
        self.save_state(state)

    def add_failed(self, config_name: str) -> None:
        """Add a config to failed list.

        Args:
            config_name: Name of failed config
        """
        state = self.load_state() or RunnerState([], [], [])
        if config_name not in state.failed_configs:
            state.failed_configs.append(config_name)
        self.save_state(state)

    def add_timedout(self, config_name: str) -> None:
        """Add a config to timedout list.

        Args:
            config_name: Name of timed out config
        """
        state = self.load_state() or RunnerState([], [], [])
        if config_name not in state.timedout_configs:
            state.timedout_configs.append(config_name)
        self.save_state(state)
