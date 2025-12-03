#!/usr/bin/env python3
"""Record baseline results from running simulation configs.

This script runs simulation configs and captures their metrics
to create lightweight baseline JSON files for CI validation.

Usage:
    # Record a single config
    python tests/scripts/record_baselines.py --config testing/femnist_bulyan_baseline.json

    # Record all fast configs (under 2 min)
    python tests/scripts/record_baselines.py --all-fast

    # List available configs without running
    python tests/scripts/record_baselines.py --list
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from tests.scripts.runner.executor import ExperimentExecutor
from tests.scripts.runner.timing_db import TimingDatabase
from rich.console import Console
from rich.table import Table

from tests.scripts.constants import FAST_CONFIGS, BASELINE_FORMAT_VERSION

# Project root for path resolution
project_root = Path(__file__).parent.parent.parent
console = Console()


def parse_round_metrics(csv_path: Path) -> dict:
    """Parse round_metrics CSV to extract all per-round metrics.

    Args:
        csv_path: Path to round_metrics_*.csv file

    Returns:
        Dictionary with per-round metrics arrays
    """
    metrics = {"total_rounds": 0, "per_round": {}}
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                metrics["total_rounds"] = len(rows)

                # Collect per-round arrays for key metrics
                per_round = metrics["per_round"]
                per_round["aggregated_loss"] = []
                per_round["average_accuracy"] = []

                for row in rows:
                    per_round["aggregated_loss"].append(
                        _safe_float(row.get("aggregated_loss_history"))
                    )
                    per_round["average_accuracy"].append(
                        _safe_float(row.get("average_accuracy_history"))
                    )

                # Final values for quick access
                final_row = rows[-1]
                metrics["final_accuracy"] = _safe_float(
                    final_row.get("average_accuracy_history")
                )
                metrics["final_loss"] = _safe_float(
                    final_row.get("aggregated_loss_history")
                )

    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse {csv_path}: {e}[/yellow]")
    return metrics


def parse_per_client_metrics(csv_path: Path, num_clients: int) -> dict:
    """Parse per_client_metrics CSV to extract all per-client metrics.

    Args:
        csv_path: Path to per_client_metrics_*.csv file
        num_clients: Number of clients to parse

    Returns:
        Dictionary with per-client metrics arrays
    """
    clients = {}
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                for client_id in range(num_clients):
                    prefix = f"client_{client_id}_"
                    clients[str(client_id)] = {
                        "loss": [],
                        "accuracy": [],
                        "removal_criterion": [],
                        "absolute_distance": [],
                        "participation": [],
                    }

                    for row in rows:
                        clients[str(client_id)]["loss"].append(
                            _safe_float(row.get(f"{prefix}loss_history"))
                        )
                        clients[str(client_id)]["accuracy"].append(
                            _safe_float(row.get(f"{prefix}accuracy_history"))
                        )
                        clients[str(client_id)]["removal_criterion"].append(
                            _safe_float(row.get(f"{prefix}removal_criterion_history"))
                        )
                        clients[str(client_id)]["absolute_distance"].append(
                            _safe_float(row.get(f"{prefix}absolute_distance_history"))
                        )
                        clients[str(client_id)]["participation"].append(
                            int(
                                _safe_float(
                                    row.get(
                                        f"{prefix}aggregation_participation_history"
                                    ),
                                    1,
                                )
                            )
                        )

    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse {csv_path}: {e}[/yellow]")
    return clients


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None or value == "" or value == "not collected":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def collect_output_info(output_dir: Path) -> dict:
    """Collect information about output files.

    Args:
        output_dir: Path to simulation output directory

    Returns:
        Dictionary with output file information
    """
    outputs = {
        "plots": [],
        "csv": [],
        "attack_snapshots": 0,
    }

    if not output_dir.exists():
        return outputs

    # Collect PDF plots
    for pdf_file in output_dir.glob("*.pdf"):
        outputs["plots"].append(pdf_file.name)

    # Collect CSV files
    csv_dir = output_dir / "csv"
    if csv_dir.exists():
        for csv_file in csv_dir.glob("*.csv"):
            outputs["csv"].append(csv_file.name)

    # Count attack snapshots
    for snapshot_dir in output_dir.glob("attack_snapshots_*"):
        if snapshot_dir.is_dir():
            outputs["attack_snapshots"] += len(list(snapshot_dir.glob("*")))

    return outputs


def extract_baseline_from_output(
    output_dir: Path, config_name: str, duration: float, num_clients: int = 10
) -> dict:
    """Extract baseline data from simulation output directory.

    Args:
        output_dir: Path to simulation output directory
        config_name: Name of the config file
        duration: Runtime in seconds
        num_clients: Number of clients in simulation

    Returns:
        Baseline dictionary
    """
    baseline = {
        "config": config_name,
        "recorded_at": datetime.now().isoformat(),
        "framework_version": "1.0.0",
        "baseline_format_version": BASELINE_FORMAT_VERSION,
        "success": True,
        "num_clients": num_clients,
        "strategies": [],
        "outputs": collect_output_info(output_dir),
        "runtime_seconds": round(duration, 1),
    }

    # Find all strategy CSVs (round_metrics_0.csv, round_metrics_1.csv, etc.)
    csv_dir = output_dir / "csv"
    if csv_dir.exists():
        strategy_csvs = sorted(csv_dir.glob("round_metrics_*.csv"))
        for idx, csv_path in enumerate(strategy_csvs):
            metrics = parse_round_metrics(csv_path)
            metrics["strategy_index"] = idx

            # Also parse per-client metrics
            per_client_csv = csv_dir / f"per_client_metrics_{idx}.csv"
            if per_client_csv.exists():
                metrics["per_client"] = parse_per_client_metrics(
                    per_client_csv, num_clients
                )

            baseline["strategies"].append(metrics)

    return baseline


def save_baseline(baseline: dict, baselines_dir: Path) -> Path:
    """Save baseline to JSON file.

    Args:
        baseline: Baseline dictionary
        baselines_dir: Directory to save baselines

    Returns:
        Path to saved baseline file
    """
    # Create baselines directory if it doesn't exist
    baselines_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from config name
    config_name = baseline["config"]
    baseline_name = config_name.replace(".json", ".baseline.json")
    baseline_path = baselines_dir / baseline_name

    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    return baseline_path


def get_num_clients_from_config(config_path: Path) -> int:
    """Read num_of_clients from config file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("shared_settings", {}).get("num_of_clients", 10)
    except Exception:
        return 10


def run_and_record(
    configs: list,
    config_subdir: str,
    baselines_dir: Path,
    timing_db: TimingDatabase,
) -> dict:
    """Run configs and record baselines.

    Args:
        configs: List of config filenames to run
        config_subdir: Subdirectory under config/simulation_strategies/
        baselines_dir: Directory to save baselines
        timing_db: Timing database for duration estimates

    Returns:
        Results summary
    """
    results = {
        "recorded": [],
        "failed": [],
        "skipped": [],
    }

    config_base = project_root / "config" / "simulation_strategies" / config_subdir
    console.print(
        f"\n[cyan]Recording baselines for {len(configs)} config(s)...[/cyan]\n"
    )

    with ExperimentExecutor(
        project_root=project_root,
        config_subdir=config_subdir,
        log_level="INFO",
        timeout=None,
        cleanup_mode="basic",
        timing_db=timing_db,
        skip_gc=True,
    ) as executor:
        for idx, config_name in enumerate(configs, start=1):
            console.print(
                f"\n[bold cyan]=== [{idx}/{len(configs)}] {config_name} ===[/bold cyan]"
            )

            # Get num_clients from config
            config_path = config_base / config_name
            num_clients = get_num_clients_from_config(config_path)

            # Run the experiment
            result = executor.run_experiment(config_name, idx, len(configs))

            if result.output_dir:
                output_path = project_root / result.output_dir
                baseline = extract_baseline_from_output(
                    output_path, config_name, result.duration, num_clients
                )

                if baseline["strategies"]:
                    baseline_path = save_baseline(baseline, baselines_dir)
                    console.print(
                        f"[green][OK] Saved baseline: {baseline_path.name}[/green]"
                    )

                    # Show summary
                    for strat in baseline["strategies"]:
                        acc = strat.get("final_accuracy", 0)
                        loss = strat.get("final_loss", 0)
                        rounds = strat.get("total_rounds", 0)
                        console.print(
                            f"  Strategy {strat['strategy_index']}: "
                            f"acc={acc:.1f}%, loss={loss:.4f}, rounds={rounds}"
                        )

                    results["recorded"].append(config_name)
                else:
                    console.print(
                        "[yellow][!] No strategy data found in output[/yellow]"
                    )
                    results["failed"].append(config_name)
            else:
                console.print("[red][X] Failed - no output directory[/red]")
                results["failed"].append(config_name)

            # Cleanup between runs
            executor.cleanup(is_last=(idx == len(configs)))

    return results


def list_configs(config_subdir: str, timing_db: TimingDatabase) -> None:
    """List available configs with timing info.

    Args:
        config_subdir: Subdirectory under config/simulation_strategies/
        timing_db: Timing database for duration estimates
    """
    config_dir = project_root / "config" / "simulation_strategies" / config_subdir

    if not config_dir.exists():
        console.print(f"[red]Config directory not found: {config_dir}[/red]")
        return

    configs = sorted(config_dir.glob("*.json"))

    table = Table(title=f"Configs in {config_subdir}/")
    table.add_column("#", style="dim")
    table.add_column("Config")
    table.add_column("Time", justify="right")
    table.add_column("Fast?", justify="center")

    for idx, config_path in enumerate(configs, start=1):
        config_name = config_path.name
        full_path = f"{config_subdir}/{config_name}"
        duration = timing_db.get_duration(full_path, device="cpu")

        if duration == float("inf"):
            time_str = "unknown"
        else:
            mins = int(duration // 60)
            secs = int(duration % 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"

        is_fast = "Y" if config_name in FAST_CONFIGS else ""

        table.add_row(str(idx), config_name, time_str, is_fast)

    console.print(table)
    console.print(f"\n[dim]Fast configs (< 2 min): {len(FAST_CONFIGS)}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Record baseline results from simulation configs"
    )
    parser.add_argument(
        "--config",
        help="Single config filename to run (e.g., femnist_bulyan_baseline.json)",
    )
    parser.add_argument(
        "--all-fast",
        action="store_true",
        help="Run all fast configs (under 2 minutes)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configs without running",
    )
    parser.add_argument(
        "--config-dir",
        default="testing",
        help="Subdirectory under config/simulation_strategies/ (default: testing)",
    )
    parser.add_argument(
        "--output-dir",
        default="tests/fixtures/baselines",
        help="Directory to save baselines (default: tests/fixtures/baselines)",
    )

    args = parser.parse_args()

    # Initialize timing database
    timing_db = TimingDatabase()

    if args.list:
        list_configs(args.config_dir, timing_db)
        return

    # Determine configs to run
    if args.all_fast:
        configs = FAST_CONFIGS
    elif args.config:
        configs = [args.config]
    else:
        console.print("[red]Please specify --config, --all-fast, or --list[/red]")
        sys.exit(1)

    # Verify configs exist
    config_base = project_root / "config" / "simulation_strategies" / args.config_dir
    missing = [c for c in configs if not (config_base / c).exists()]
    if missing:
        console.print(f"[red]Missing configs: {missing}[/red]")
        sys.exit(1)

    baselines_dir = project_root / args.output_dir

    # Run and record
    results = run_and_record(configs, args.config_dir, baselines_dir, timing_db)

    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]Recording Summary[/bold cyan]")
    console.print("=" * 50)
    console.print(f"[green]Recorded: {len(results['recorded'])}[/green]")
    if results["failed"]:
        console.print(f"[red]Failed: {len(results['failed'])}[/red]")
        for config in results["failed"]:
            console.print(f"  - {config}")

    console.print(f"\n[cyan]Baselines saved to: {baselines_dir}[/cyan]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
