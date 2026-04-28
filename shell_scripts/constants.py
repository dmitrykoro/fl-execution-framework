"""Shared test constants across CI scripts.

This module centralizes configuration lists used by:
- ci_smoke_test.py
- mock_simulation_runner.py
- record_baselines.py

It dynamically loads all configurations in the CPU directory.
"""

from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CPU_CONFIGS_DIR = PROJECT_ROOT / "config" / "simulation_strategies" / "cpu"

def get_all_configs() -> list[str]:
    """Get all valid configs in config/simulation_strategies/cpu/ directory."""
    if not CPU_CONFIGS_DIR.exists():
        return []
    return sorted([f.name for f in CPU_CONFIGS_DIR.glob("*.json")])

ALL_CONFIGS = get_all_configs()
BASELINE_CONFIGS = [c for c in ALL_CONFIGS if "baseline" in c]
ATTACK_CONFIGS = [c for c in ALL_CONFIGS if "vs_" in c or "multi_" in c]

assert len(ALL_CONFIGS) == len(set(ALL_CONFIGS)), "ALL_CONFIGS contains duplicates!"

BASELINE_FORMAT_VERSION = "1.0.0"
