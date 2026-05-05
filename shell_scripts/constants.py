"""Shared constants for the smoke-test machinery.

Centralizes configuration discovery used by:
- smoke_test.py
- mock_simulation_runner.py
- record_baselines.py

Auto-discovers every JSON config in config/simulation_strategies/smoke/.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SMOKE_CONFIGS_DIR = PROJECT_ROOT / "config" / "simulation_strategies" / "smoke"


def get_all_configs() -> list[str]:
    """Return every config in the smoke directory, sorted alphabetically."""
    if not SMOKE_CONFIGS_DIR.exists():
        return []
    return sorted([f.name for f in SMOKE_CONFIGS_DIR.glob("*.json")])


SMOKE_CONFIGS = get_all_configs()
