#!/usr/bin/env python3
"""CI smoke test - run mock simulations and verify output generation.

This script runs the full simulation pipeline using mock training data
to test that all output generation code works correctly.

Usage:
    # Default: run mock simulations for all fast configs
    python tests/scripts/ci_smoke_test.py

    # Run single config
    python tests/scripts/ci_smoke_test.py --config femnist_krum_baseline.json

    # Parse-only mode (faster, just validates config syntax)
    python tests/scripts/ci_smoke_test.py --parse-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console

console = Console()

# Same fast configs as other scripts
FAST_CONFIGS = [
    "breastmnist_krum_vs_labelflip20.json",
    "femnist_pidstdscore_baseline.json",
    "femnist_bulyan_baseline.json",
    "femnist_mkrum_baseline.json",
    "femnist_trust_baseline.json",
    "femnist_rfa_baseline.json",
    "femnist_pidstd_baseline.json",
    "femnist_trimmean_baseline.json",
    "femnist_pid_baseline.json",
    "femnist_pidscaled_baseline.json",
    "femnist_krum_baseline.json",
    "femnist_rfa_vs_labelflip20.json",
    "femnist_mkrum_vs_labelflip20.json",
    "femnist_krum_multi_overlapping.json",
    "femnist_trust_vs_labelflip20.json",
    "femnist_pidstd_vs_labelflip20.json",
    "femnist_krum_vs_labelflip20.json",
    "femnist_mkrum_vs_labelflip50.json",
    "femnist_pidstdscore_vs_labelflip20.json",
    "femnist_bulyan_vs_labelflip50.json",
    "femnist_mkrum_multi_concurrent.json",
    "femnist_krum_vs_labelflip50.json",
    "femnist_mkrum_multi_showcase.json",
    "femnist_mkrum_vs_gaussnoise25.json",
]


def validate_config_parsing(config_path: Path) -> tuple[bool, list[str]]:
    """Validate that a config file parses correctly."""
    errors = []

    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}")
        return False, errors

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors

    if "shared_settings" not in config:
        errors.append("Missing 'shared_settings' field")

    if "simulation_strategies" not in config:
        errors.append("Missing 'simulation_strategies' field")

    shared = config.get("shared_settings", {})
    for field in ["aggregation_strategy_keyword", "num_of_rounds", "num_of_clients"]:
        if field not in shared:
            errors.append(f"Missing required field: shared_settings.{field}")

    return len(errors) == 0, errors


def load_baseline(config_name: str, baselines_dir: Path) -> dict | None:
    """Load baseline data for a config."""
    baseline_name = config_name.replace(".json", ".baseline.json")
    baseline_path = baselines_dir / baseline_name

    if not baseline_path.exists():
        return None

    with open(baseline_path) as f:
        return json.load(f)


def run_parse_only(configs: list[str], config_dir: Path) -> int:
    """Run parse-only validation (fast mode)."""
    console.print(f"[cyan]Validating {len(configs)} config(s) (parse-only)...[/cyan]\n")

    passed = 0
    failed = 0

    for config in configs:
        config_path = config_dir / config
        ok, errors = validate_config_parsing(config_path)

        if ok:
            console.print(f"  [green]OK[/green] {config}")
            passed += 1
        else:
            console.print(f"  [red]X[/red] {config}")
            for err in errors:
                console.print(f"      {err}")
            failed += 1

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    return 0 if failed == 0 else 1


def run_mock_simulations(configs: list[str], config_dir: Path, baselines_dir: Path) -> int:
    """Run mock simulations with full output generation."""
    # Import mock runner (deferred to avoid heavy imports in parse-only mode)
    from tests.scripts.mock_simulation_runner import run_mock_simulation, verify_outputs

    # Check which configs have baselines
    configs_with_baselines = []
    configs_without = []

    for config in configs:
        if load_baseline(config, baselines_dir):
            configs_with_baselines.append(config)
        else:
            configs_without.append(config)

    if configs_without:
        console.print(f"[yellow]Missing baselines for {len(configs_without)} config(s)[/yellow]")
        console.print("[yellow]Falling back to parse-only for those.[/yellow]\n")

    if configs_with_baselines:
        console.print(f"[cyan]Running {len(configs_with_baselines)} mock simulation(s)...[/cyan]\n")

    passed = 0
    failed = 0

    # Run mock simulations for configs with baselines
    for idx, config in enumerate(configs_with_baselines, start=1):
        console.print(f"[{idx}/{len(configs_with_baselines)}] {config}...", end=" ")

        success, output_dir, errors = run_mock_simulation(
            config, config_dir, baselines_dir, project_root / "out"
        )

        if success and output_dir:
            verify_ok, verify_errors = verify_outputs(output_dir)
            if verify_ok:
                console.print("[green]OK[/green]")
                passed += 1
            else:
                console.print("[red]VERIFY FAILED[/red]")
                for err in verify_errors:
                    console.print(f"  - {err}")
                failed += 1
        else:
            console.print("[red]FAILED[/red]")
            for err in errors[:3]:
                console.print(f"  - {err}")
            failed += 1

    # Parse-only for configs without baselines
    for config in configs_without:
        config_path = config_dir / config
        ok, errors = validate_config_parsing(config_path)
        if ok:
            console.print(f"  [green]OK[/green] {config} (parse-only)")
            passed += 1
        else:
            console.print(f"  [red]X[/red] {config}")
            failed += 1

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="CI smoke test - run mock simulations and verify outputs"
    )
    parser.add_argument(
        "--config",
        help="Single config filename to test",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only validate config parsing (faster, no output generation)",
    )
    parser.add_argument(
        "--config-dir",
        default="testing",
        help="Subdirectory under config/simulation_strategies/ (default: testing)",
    )
    parser.add_argument(
        "--baselines-dir",
        default="tests/fixtures/baselines",
        help="Directory containing baselines (default: tests/fixtures/baselines)",
    )

    args = parser.parse_args()

    configs = [args.config] if args.config else FAST_CONFIGS
    config_dir = project_root / "config" / "simulation_strategies" / args.config_dir
    baselines_dir = project_root / args.baselines_dir

    # Verify configs exist
    missing = [c for c in configs if not (config_dir / c).exists()]
    if missing:
        console.print(f"[red]Missing configs: {missing}[/red]")
        sys.exit(1)

    if args.parse_only:
        exit_code = run_parse_only(configs, config_dir)
    else:
        exit_code = run_mock_simulations(configs, config_dir, baselines_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
