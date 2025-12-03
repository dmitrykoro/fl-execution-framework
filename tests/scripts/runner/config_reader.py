"""Config file reader utility for experiments."""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ConfigMetadata:
    """Metadata extracted from a config file."""

    title: Optional[str]
    device: str
    num_rounds: int
    num_clients: int

    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return self.title if self.title else "Untitled Experiment"


class ConfigReader:
    """Utility for reading experiment configuration files."""

    def __init__(self, project_root: Path):
        """Initialize config reader.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.config_dir = (
            self.project_root / "config" / "simulation_strategies" / "testing"
        )
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _get_config_path(self, config_name: str) -> Path:
        """Get full path to config file.

        Args:
            config_name: Name of config file

        Returns:
            Full path to config file
        """
        return self.config_dir / config_name

    def _load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load config file as dictionary.

        Args:
            config_name: Name of config file
            use_cache: Whether to use cached config (default: True)

        Returns:
            Config dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is not valid JSON
        """
        if use_cache and config_name in self._cache:
            return self._cache[config_name]

        config_path = self._get_config_path(config_name)

        with open(config_path, "r") as f:
            config = json.load(f)

        if use_cache:
            self._cache[config_name] = config

        return config

    def get_title(self, config_name: str) -> Optional[str]:
        """Extract title from config file.

        Args:
            config_name: Name of config file

        Returns:
            Title string if found, None otherwise
        """
        try:
            config = self._load_config(config_name)
            return config.get("_title")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def get_device(self, config_name: str) -> str:
        """Extract training device from config file.

        Args:
            config_name: Name of config file

        Returns:
            Device string ("cuda", "cpu", "auto"), defaults to "cuda"
        """
        try:
            config = self._load_config(config_name)

            # Check shared_settings first, then root level
            if "shared_settings" in config:
                return config["shared_settings"].get("training_device", "cuda")
            return config.get("training_device", "cuda")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return "cuda"

    def get_metadata(self, config_name: str) -> ConfigMetadata:
        """Extract metadata from config file.

        Args:
            config_name: Name of config file

        Returns:
            ConfigMetadata object with extracted information
        """
        try:
            config = self._load_config(config_name)
            shared = config.get("shared_settings", {})

            return ConfigMetadata(
                title=config.get("_title"),
                device=shared.get("training_device", "cuda"),
                num_rounds=shared.get("num_of_rounds", 0),
                num_clients=shared.get("num_of_clients", 0),
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return ConfigMetadata(
                title=None, device="cuda", num_rounds=0, num_clients=0
            )

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()
