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
import csv
import json
import sys
from pathlib import Path
from rich.console import Console

from tests.scripts.constants import FAST_CONFIGS, BASELINE_FORMAT_VERSION

# Project root for path resolution
project_root = Path(__file__).parent.parent.parent
console = Console()


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


def extract_metrics_from_csv(output_dir: Path) -> dict[int, dict]:
    """Extract final metrics from generated CSV files.

    Args:
        output_dir: Path to simulation output directory

    Returns:
        Dictionary mapping strategy index to metrics dict with keys:
        - final_accuracy: Last round's average accuracy
        - final_loss: Last round's aggregated loss
        - total_rounds: Number of rounds
    """
    metrics = {}
    csv_dir = output_dir / "csv"

    if not csv_dir.exists():
        return metrics

    for csv_file in sorted(csv_dir.glob("round_metrics_*.csv")):
        # Extract strategy index from filename (round_metrics_0.csv -> 0)
        try:
            strategy_idx = int(csv_file.stem.split("_")[-1])
        except ValueError:
            continue

        try:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if rows:
                    final_row = rows[-1]
                    metrics[strategy_idx] = {
                        "final_accuracy": _safe_float(
                            final_row.get("average_accuracy_history")
                        ),
                        "final_loss": _safe_float(
                            final_row.get("aggregated_loss_history")
                        ),
                        "total_rounds": len(rows),
                    }
        except Exception:
            continue

    return metrics


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None or value == "" or value == "not collected":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def compare_metrics(
    baseline: dict, current_metrics: dict[int, dict], tolerance: float = 0.01
) -> list[str]:
    """Compare current run metrics against baseline.

    Since mock simulation uses baseline data to populate history,
    output metrics should match baseline closely. Deviations indicate
    bugs in populate_history_from_baseline() or calculate_additional_rounds_data().

    Args:
        baseline: Recorded baseline data
        current_metrics: Metrics extracted from current mock run
        tolerance: Acceptable relative deviation (1% default - should be nearly exact)

    Returns:
        List of warnings if metrics deviate significantly
    """
    warnings = []

    for strategy in baseline.get("strategies", []):
        idx = strategy.get("strategy_index", 0)
        baseline_acc = strategy.get("final_accuracy", 0)
        baseline_loss = strategy.get("final_loss", 0)
        baseline_rounds = strategy.get("total_rounds", 0)

        current = current_metrics.get(idx, {})
        current_acc = current.get("final_accuracy", 0)
        current_loss = current.get("final_loss", 0)
        current_rounds = current.get("total_rounds", 0)

        # Check rounds match
        if current_rounds != baseline_rounds:
            warnings.append(
                f"Strategy {idx}: round count mismatch "
                f"(expected {baseline_rounds}, got {current_rounds})"
            )

        # Check accuracy (relative tolerance)
        if baseline_acc > 0:
            acc_diff = abs(current_acc - baseline_acc) / baseline_acc
            if acc_diff > tolerance:
                warnings.append(
                    f"Strategy {idx}: accuracy deviation {acc_diff:.1%} "
                    f"(expected {baseline_acc:.2f}, got {current_acc:.2f})"
                )

        # Check loss (relative tolerance)
        if baseline_loss > 0:
            loss_diff = abs(current_loss - baseline_loss) / baseline_loss
            if loss_diff > tolerance:
                warnings.append(
                    f"Strategy {idx}: loss deviation {loss_diff:.1%} "
                    f"(expected {baseline_loss:.4f}, got {current_loss:.4f})"
                )

    return warnings


def check_baseline_staleness(baseline: dict, config_name: str) -> list[str]:
    """Check if baseline may be stale.

    Args:
        baseline: Loaded baseline data
        config_name: Name of config for error messages

    Returns:
        List of warnings about staleness
    """
    warnings = []

    # Check format version
    baseline_version = baseline.get("baseline_format_version")
    if baseline_version and baseline_version != BASELINE_FORMAT_VERSION:
        warnings.append(
            f"{config_name}: baseline format version mismatch "
            f"(baseline: {baseline_version}, current: {BASELINE_FORMAT_VERSION})"
        )

    # Check age (warn if older than 60 days)
    recorded_at = baseline.get("recorded_at")
    if recorded_at:
        try:
            from datetime import datetime

            recorded = datetime.fromisoformat(recorded_at)
            age_days = (datetime.now() - recorded).days
            if age_days > 60:
                warnings.append(
                    f"{config_name}: baseline is {age_days} days old, consider re-recording"
                )
        except Exception:
            pass

    return warnings


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


def run_mock_simulations(
    configs: list[str], config_dir: Path, baselines_dir: Path, verbose: bool = False
) -> int:
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
        console.print(
            f"[yellow]Missing baselines for {len(configs_without)} config(s)[/yellow]"
        )
        console.print("[yellow]Falling back to parse-only for those.[/yellow]\n")

    if configs_with_baselines:
        console.print(
            f"[cyan]Running {len(configs_with_baselines)} mock simulation(s)...[/cyan]\n"
        )

    passed = 0
    failed = 0
    staleness_warnings = []
    metric_warnings = []
    results = []  # Track results for summary table

    # Run mock simulations for configs with baselines
    for idx, config in enumerate(configs_with_baselines, start=1):
        console.print(f"[{idx}/{len(configs_with_baselines)}] {config}...", end=" ")

        # Load baseline for comparison
        baseline = load_baseline(config, baselines_dir)

        # Check baseline staleness
        if baseline:
            stale_warns = check_baseline_staleness(baseline, config)
            staleness_warnings.extend(stale_warns)

        success, output_dir, errors = run_mock_simulation(
            config, config_dir, baselines_dir, project_root / "out"
        )

        if success and output_dir:
            verify_ok, verify_errors, counts = verify_outputs(output_dir)

            # Compare metrics against baseline
            metric_warns = []
            if verify_ok and baseline:
                current_metrics = extract_metrics_from_csv(output_dir)
                metric_warns = compare_metrics(baseline, current_metrics)
                metric_warnings.extend([(config, w) for w in metric_warns])

            if verify_ok and not metric_warns:
                console.print("[green]OK[/green]")
                passed += 1
                results.append(
                    {
                        "config": config,
                        "status": "OK",
                        "plots": counts["plots"],
                        "csvs": counts["csvs"],
                        "snapshots": counts["snapshots"],
                        "output_dir": str(output_dir) if verbose else None,
                    }
                )
            elif verify_ok and metric_warns:
                console.print("[yellow]OK (metric warnings)[/yellow]")
                passed += 1  # Still pass, but with warnings
                results.append(
                    {
                        "config": config,
                        "status": "OK*",
                        "plots": counts["plots"],
                        "csvs": counts["csvs"],
                        "snapshots": counts["snapshots"],
                        "output_dir": str(output_dir) if verbose else None,
                    }
                )
            else:
                console.print("[red]VERIFY FAILED[/red]")
                for err in verify_errors:
                    console.print(f"  - {err}")
                failed += 1
                results.append(
                    {
                        "config": config,
                        "status": "VERIFY FAILED",
                        "plots": counts["plots"],
                        "csvs": counts["csvs"],
                        "snapshots": counts["snapshots"],
                        "output_dir": None,
                    }
                )
        else:
            console.print("[red]FAILED[/red]")
            for err in errors[:3]:
                console.print(f"  - {err}")
            failed += 1
            results.append(
                {
                    "config": config,
                    "status": "FAILED",
                    "plots": 0,
                    "csvs": 0,
                    "snapshots": 0,
                    "output_dir": None,
                }
            )

    # Parse-only for configs without baselines
    for config in configs_without:
        config_path = config_dir / config
        ok, errors = validate_config_parsing(config_path)
        if ok:
            console.print(f"  [green]OK[/green] {config} (parse-only)")
            passed += 1
            results.append(
                {
                    "config": config,
                    "status": "OK (parse)",
                    "plots": "-",
                    "csvs": "-",
                    "snapshots": "-",
                    "output_dir": None,
                }
            )
        else:
            console.print(f"  [red]X[/red] {config}")
            failed += 1

    # Print summary table
    _print_summary_table(results, verbose)

    # Print staleness warnings
    if staleness_warnings:
        console.print("\n[yellow]Baseline Staleness Warnings:[/yellow]")
        for warn in staleness_warnings:
            console.print(f"  [yellow]![/yellow] {warn}")

    # Print metric warnings
    if metric_warnings:
        console.print("\n[yellow]Metric Regression Warnings:[/yellow]")
        for config, warn in metric_warnings:
            console.print(f"  [yellow]![/yellow] {config}: {warn}")

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    if metric_warnings:
        console.print(
            f"[dim]  ({len(metric_warnings)} metric warning(s) - "
            "may indicate changes in output logic)[/dim]"
        )
    return 0 if failed == 0 else 1


def _print_summary_table(results: list[dict], verbose: bool = False) -> None:
    """Print a summary table of all test results."""
    if not results:
        return

    console.print("\n")

    # Calculate column widths
    max_config_len = max(len(r["config"]) for r in results)
    config_width = min(max_config_len, 45)

    # Header
    header = f"{'Config':<{config_width}} | Plots | CSVs | Snaps | Status"
    console.print(f"[bold]{header}[/bold]")
    console.print("-" * len(header))

    # Rows
    for r in results:
        config_name = r["config"]
        if len(config_name) > config_width:
            config_name = config_name[: config_width - 3] + "..."

        plots = str(r["plots"]).center(5)
        csvs = str(r["csvs"]).center(4)
        snaps = str(r["snapshots"]).center(5)

        status_color = "green" if r["status"] == "OK" else "red"
        if r["status"] in ("OK (parse)", "OK*"):
            status_color = "yellow"

        console.print(
            f"{config_name:<{config_width}} | {plots} | {csvs} | {snaps} | "
            f"[{status_color}]{r['status']}[/{status_color}]"
        )

        if verbose and r.get("output_dir"):
            console.print(f"  [dim]Output: {r['output_dir']}[/dim]")


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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output including file paths",
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
        exit_code = run_mock_simulations(
            configs, config_dir, baselines_dir, args.verbose
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
