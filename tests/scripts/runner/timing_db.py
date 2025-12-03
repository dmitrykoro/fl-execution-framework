"""Timing database for tracking experiment execution times."""

import json
from pathlib import Path
from typing import Optional


class TimingDatabase:
    """Track and persist experiment execution times."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize timing database.

        Args:
            db_path: Path to timing database JSON file.
                     Defaults to timing_db.json in project root
        """
        if db_path is None:
            # Store timing DB in project root
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = project_root / "timing_db.json"

        self.db_path = Path(db_path)
        self.timings = self._load()

    def _load(self) -> dict:
        """Load timing data from JSON file.

        Returns:
            Dictionary mapping config_path -> {device -> duration_seconds}
            where config_path includes subdirectory (e.g., "examples/test.json")
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or unreadable, start fresh
                return {}
        return {}

    def record(self, config_name: str, duration: float, device: str = "gpu") -> None:
        """Record execution time for a config on a specific device.

        Tracks the fastest (minimum) execution time and total run count.
        This provides a stable metric for test suite optimization while
        maintaining confidence through run tracking.

        Args:
            config_name: Path to config file including subdirectory (e.g., "examples/test.json")
            duration: Execution time in seconds
            device: Device type ("gpu", "cpu")
        """
        if config_name not in self.timings:
            self.timings[config_name] = {}

        rounded_duration = round(duration, 1)

        if device not in self.timings[config_name]:
            # First run for this device
            self.timings[config_name][device] = {"time": rounded_duration, "runs": 1}
        else:
            current = self.timings[config_name][device]
            # Handle backward compatibility: convert old format to new
            if isinstance(current, (int, float)):
                self.timings[config_name][device] = {
                    "time": min(current, rounded_duration),
                    "runs": 2,
                }
            else:
                # Update fastest time if this run is faster
                if rounded_duration < current["time"]:
                    current["time"] = rounded_duration
                current["runs"] += 1

        self._save()

    def _save(self) -> None:
        """Persist timing data to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            # Sort by config name for better readability
            sorted_timings = dict(sorted(self.timings.items()))
            json.dump(sorted_timings, f, indent=2)

    def get_duration(
        self, config_name: str, device: str = "gpu", default: float = float("inf")
    ) -> float:
        """Get recorded duration for a config on a specific device.

        Returns the fastest (minimum) recorded time for stable test ordering.

        Args:
            config_name: Path to config file including subdirectory (e.g., "examples/test.json")
            device: Device type ("gpu", "cpu")
            default: Default value if config not found (default: infinity)

        Returns:
            Duration in seconds, or default if not found
        """
        if config_name not in self.timings:
            return default

        device_data = self.timings[config_name].get(device)
        if device_data is None:
            return default

        # Handle backward compatibility: old format was just a number
        if isinstance(device_data, (int, float)):
            return device_data

        # New format: nested dictionary with "time" key
        return device_data.get("time", default)

    def has_timing(self, config_name: str, device: str = "gpu") -> bool:
        """Check if timing exists for a config on a specific device.

        Args:
            config_name: Path to config file including subdirectory (e.g., "examples/test.json")
            device: Device type ("gpu", "cpu")

        Returns:
            True if timing recorded, False otherwise
        """
        return config_name in self.timings and device in self.timings[config_name]

    def get_stats(self, device: Optional[str] = None) -> dict:
        """Get summary statistics, optionally filtered by device.

        Args:
            device: Optional device filter ("gpu", "cpu"). None = all devices.

        Returns:
            Dictionary with count, min, max, avg timing stats
        """
        # Collect all durations (filtered by device if specified)
        durations = []
        for config_timings in self.timings.values():
            if device is None:
                # All devices
                for device_data in config_timings.values():
                    # Handle backward compatibility
                    if isinstance(device_data, (int, float)):
                        durations.append(device_data)
                    else:
                        durations.append(device_data["time"])
            elif device in config_timings:
                # Specific device only
                device_data = config_timings[device]
                # Handle backward compatibility
                if isinstance(device_data, (int, float)):
                    durations.append(device_data)
                else:
                    durations.append(device_data["time"])

        if not durations:
            return {"count": 0, "min": 0, "max": 0, "avg": 0}

        return {
            "count": len(durations),
            "min": round(min(durations), 1),
            "max": round(max(durations), 1),
            "avg": round(sum(durations) / len(durations), 1),
        }

    def clear(self) -> None:
        """Clear all timing data."""
        self.timings = {}
        if self.db_path.exists():
            self.db_path.unlink()
