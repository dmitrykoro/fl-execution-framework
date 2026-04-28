"""Timing database for tracking experiment execution times."""

from __future__ import annotations

import json
from pathlib import Path


class TimingDatabase:
    """Track and persist experiment execution times."""

    def __init__(self, db_path: Path | None = None):
        """Initialize timing database.

        Args:
            db_path: Path to timing database JSON file.
        """
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "timing_db.json"

        self.db_path = Path(db_path)
        self.timings = self._load()

    def _load(self) -> dict:
        """Load timing data from JSON file.

        Returns:
            Dictionary mapping config_name to device timings.
        """
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def record(self, config_name: str, duration: float, device: str = "gpu") -> None:
        """Record execution time for a config on a specific device.

        Args:
            config_name: Config filename.
            duration: Execution time in seconds.
            device: Device type ("gpu" or "cpu").
        """
        if config_name not in self.timings:
            self.timings[config_name] = {}

        rounded_duration = round(duration, 1)

        if device not in self.timings[config_name]:
            self.timings[config_name][device] = {"time": rounded_duration, "runs": 1}
        else:
            current = self.timings[config_name][device]
            if isinstance(current, (int, float)):
                self.timings[config_name][device] = {
                    "time": min(current, rounded_duration),
                    "runs": 2,
                }
            else:
                if rounded_duration < current["time"]:
                    current["time"] = rounded_duration
                current["runs"] += 1

        self._save()

    def _save(self) -> None:
        """Persist timing data to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            sorted_timings = dict(sorted(self.timings.items()))
            json.dump(sorted_timings, f, indent=2)

    def get_duration(
        self, config_name: str, device: str = "gpu", default: float = float("inf")
    ) -> float:
        """Get recorded duration for a config on a specific device.

        Args:
            config_name: Config filename.
            device: Device type ("gpu" or "cpu").
            default: Default value if config not found.

        Returns:
            Duration in seconds, or default if not found.
        """
        if config_name not in self.timings:
            return default

        device_data = self.timings[config_name].get(device)
        if device_data is None:
            return default

        if isinstance(device_data, (int, float)):
            return device_data

        return device_data.get("time", default)

    def has_timing(self, config_name: str, device: str = "gpu") -> bool:
        """Check if timing exists for a config on a specific device.

        Args:
            config_name: Config filename.
            device: Device type ("gpu" or "cpu").

        Returns:
            True if timing recorded, False otherwise.
        """
        return config_name in self.timings and device in self.timings[config_name]

    def get_stats(self, device: str | None = None) -> dict:
        """Get summary statistics, optionally filtered by device.

        Args:
            device: Optional device filter ("gpu" or "cpu").

        Returns:
            Dictionary with count, min, max, avg timing stats.
        """
        durations = []
        for config_timings in self.timings.values():
            if device is None:
                for device_data in config_timings.values():
                    if isinstance(device_data, (int, float)):
                        durations.append(device_data)
                    else:
                        durations.append(device_data["time"])
            elif device in config_timings:
                device_data = config_timings[device]
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
