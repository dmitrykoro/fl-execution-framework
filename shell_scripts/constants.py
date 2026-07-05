"""Smoke-test config discovery."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SMOKE_CONFIGS_DIR = PROJECT_ROOT / "config" / "simulation_strategies" / "smoke"
BASELINES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "baselines"
SMOKE_CONFIGS = sorted(p.name for p in SMOKE_CONFIGS_DIR.glob("*.json"))
