#!/usr/bin/env python3
"""Mock simulation runner - runs full output generation using recorded baselines.

This script runs the complete simulation pipeline (plots, CSVs, HTML reports)
using pre-recorded baseline data instead of actual Flower training.

Usage:
    # Run single config with mock data
    python tests/scripts/mock_simulation_runner.py --config femnist_krum_baseline.json

    # Run all fast configs
    python tests/scripts/mock_simulation_runner.py --all-fast
"""

import argparse
import csv
import json
import logging
import sys
import traceback
import torch
from pathlib import Path
from rich.console import Console
from tests.scripts.constants import FAST_CONFIGS
from src.attack_utils.attack_snapshots import save_attack_snapshot
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.output_handlers import new_plot_handler
from src.output_handlers.directory_handler import DirectoryHandler
from src.attack_utils.snapshot_html_reports import (
    generate_main_dashboard,
    generate_snapshot_index,
    generate_summary_json,
)

# Project root for path resolution
project_root = Path(__file__).parent.parent.parent
console = Console()


def load_baseline(config_name: str, baselines_dir: Path) -> dict | None:
    """Load baseline data for a config."""
    baseline_name = config_name.replace(".json", ".baseline.json")
    baseline_path = baselines_dir / baseline_name

    if not baseline_path.exists():
        return None

    with open(baseline_path) as f:
        return json.load(f)


def load_config(config_name: str, config_dir: Path) -> dict:
    """Load simulation config."""
    config_path = config_dir / config_name
    with open(config_path) as f:
        return json.load(f)


def populate_history_from_baseline(
    strategy_history, baseline_strategy: dict, num_clients: int
):
    """Populate SimulationStrategyHistory with baseline data."""
    per_round = baseline_strategy.get("per_round", {})
    per_client = baseline_strategy.get("per_client", {})
    total_rounds = baseline_strategy.get("total_rounds", 10)

    # Populate per-client data
    for client_id in range(num_clients):
        client_data = per_client.get(str(client_id), {})
        for round_num in range(1, total_rounds + 1):
            idx = round_num - 1  # 0-indexed

            strategy_history.insert_single_client_history_entry(
                client_id=client_id,
                current_round=round_num,
                removal_criterion=client_data.get(
                    "removal_criterion", [0.0] * total_rounds
                )[idx],
                absolute_distance=client_data.get(
                    "absolute_distance", [0.0] * total_rounds
                )[idx],
                loss=client_data.get("loss", [0.0] * total_rounds)[idx],
                accuracy=client_data.get("accuracy", [0.0] * total_rounds)[idx],
                aggregation_participation=client_data.get(
                    "participation", [1] * total_rounds
                )[idx],
            )

    # Populate per-round data
    aggregated_loss = per_round.get("aggregated_loss", [0.0] * total_rounds)
    for round_num in range(1, total_rounds + 1):
        idx = round_num - 1
        strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=0,
            removal_threshold=0.0,
            loss_aggregated=aggregated_loss[idx] if idx < len(aggregated_loss) else 0.0,
        )


class MockFederatedSimulation:
    """Mock simulation that uses baseline data instead of real training."""

    def __init__(self, strategy_config, strategy_history, attack_schedule=None):
        self.strategy_config = strategy_config
        self.strategy_history = strategy_history
        self._attack_schedule = attack_schedule or []

    def get_attack_schedule_as_dict(self):
        return self._attack_schedule


def generate_mock_attack_snapshots(
    attack_schedule: list,
    output_dir: str,
    num_clients: int,
    total_rounds: int,
    strategy_number: int,
    max_samples: int = 5,
    save_format: str = "pickle",
) -> int:
    """Generate mock attack snapshots for CI testing.

    Creates synthetic snapshot data for attacks defined in the attack_schedule.
    This allows testing the snapshot generation pipeline without actual training.

    Args:
        attack_schedule: List of attack configurations
        output_dir: Output directory path
        num_clients: Total number of clients
        total_rounds: Total number of rounds
        strategy_number: Strategy index
        max_samples: Max samples per snapshot
        save_format: Snapshot format (pickle, visual, pickle_and_visual)

    Returns:
        Number of snapshots generated
    """
    snapshots_generated = 0

    for attack in attack_schedule:
        attack_type = attack.get("attack_type", "unknown")
        start_round = attack.get("start_round", 1)
        end_round = attack.get("end_round", total_rounds)

        # Determine malicious clients
        selection = attack.get("selection_strategy", "percentage")
        if selection == "percentage":
            percentage = attack.get("malicious_percentage", 0.2)
            num_malicious = max(1, int(num_clients * percentage))
            malicious_clients = list(range(num_malicious))
        elif selection == "specific":
            malicious_clients = attack.get("client_ids", [0])
        else:
            malicious_clients = [0]

        # Generate snapshots for each affected client/round
        for round_num in range(start_round, min(end_round, total_rounds) + 1):
            for client_id in malicious_clients:
                # Create mock tensor data (28x28 grayscale images like FEMNIST)
                mock_data = torch.rand(max_samples, 1, 28, 28)
                mock_labels = torch.randint(0, 10, (max_samples,))

                # For label flipping, create original labels different from current
                if attack_type == "label_flipping":
                    original_labels = (mock_labels + 1) % 10
                else:
                    original_labels = mock_labels.clone()

                try:
                    save_attack_snapshot(
                        client_id=client_id,
                        round_num=round_num,
                        attack_config=attack,
                        data_sample=mock_data,
                        labels_sample=mock_labels,
                        original_labels_sample=original_labels,
                        output_dir=output_dir,
                        max_samples=max_samples,
                        save_format=save_format,
                        strategy_number=strategy_number,
                    )
                    snapshots_generated += 1
                except Exception as e:
                    logging.warning(
                        f"Failed to generate mock snapshot for client {client_id}, "
                        f"round {round_num}: {e}"
                    )

    return snapshots_generated


def run_mock_simulation(
    config_name: str,
    config_dir: Path,
    baselines_dir: Path,
    output_base: Path,
) -> tuple[bool, Path | None, list[str]]:
    """Run mock simulation using baseline data.

    Returns:
        Tuple of (success, output_dir, errors)
    """
    errors = []

    # Load baseline
    baseline = load_baseline(config_name, baselines_dir)
    if not baseline:
        errors.append(f"No baseline found for {config_name}")
        return False, None, errors

    # Load config
    try:
        config = load_config(config_name, config_dir)
    except Exception as e:
        errors.append(f"Failed to load config: {e}")
        return False, None, errors

    try:
        # Create directory handler for output
        directory_handler = DirectoryHandler()
        output_dir = Path(directory_handler.dirname)

        # Process each strategy in the config
        shared_settings = config.get("shared_settings", {})
        strategies = config.get("simulation_strategies", [{}])
        num_clients = baseline.get(
            "num_clients", shared_settings.get("num_of_clients", 10)
        )

        executed_simulations = []

        for strat_idx, strategy_overrides in enumerate(strategies):
            # Merge config
            merged_config = {**shared_settings, **strategy_overrides}
            merged_config["strategy_number"] = strat_idx

            # Get baseline for this strategy
            if strat_idx < len(baseline.get("strategies", [])):
                baseline_strategy = baseline["strategies"][strat_idx]
            else:
                baseline_strategy = (
                    baseline["strategies"][0] if baseline.get("strategies") else {}
                )

            # Create strategy config
            strategy_config = StrategyConfig.from_dict(merged_config)
            setattr(strategy_config, "strategy_number", strat_idx)

            # Create mock dataset handler (minimal, just for initialization)
            class MockDatasetHandler:
                def __init__(self):
                    self.malicious_clients = set()
                    # Parse attack schedule to find malicious clients
                    for attack in merged_config.get("attack_schedule", []):
                        selected = attack.get("_selected_clients", [])
                        self.malicious_clients.update(selected)

            dataset_handler = MockDatasetHandler()

            # Create strategy history
            strategy_history = SimulationStrategyHistory(
                strategy_config=strategy_config,
                dataset_handler=dataset_handler,  # type: ignore[arg-type]
            )

            # Populate from baseline
            populate_history_from_baseline(
                strategy_history, baseline_strategy, num_clients
            )

            # Calculate additional metrics
            strategy_history.calculate_additional_rounds_data()

            # Create mock simulation object for plotting
            mock_sim = MockFederatedSimulation(
                strategy_config=strategy_config,
                strategy_history=strategy_history,
                attack_schedule=merged_config.get("attack_schedule", []),
            )

            # Generate plots
            if strategy_config.save_plots:
                new_plot_handler.show_plots_within_strategy(
                    mock_sim,  # pyright: ignore[reportArgumentType]
                    directory_handler,
                )

            # Save CSVs
            if strategy_config.save_csv:
                directory_handler.save_csv_and_config(strategy_history)

            # Generate attack snapshots if attack_schedule exists
            attack_schedule = merged_config.get("attack_schedule", [])
            save_snapshots = merged_config.get("save_attack_snapshots", "false")
            if attack_schedule and str(save_snapshots).lower() == "true":
                total_rounds = baseline_strategy.get(
                    "total_rounds", merged_config.get("num_of_rounds", 10)
                )
                snapshot_format = merged_config.get("attack_snapshot_format", "pickle")
                max_samples = merged_config.get("snapshot_max_samples", 5)

                generate_mock_attack_snapshots(
                    attack_schedule=attack_schedule,
                    output_dir=directory_handler.dirname,
                    num_clients=num_clients,
                    total_rounds=total_rounds,
                    strategy_number=strat_idx,
                    max_samples=max_samples,
                    save_format=snapshot_format,
                )

                # Generate snapshot index and summary
                try:
                    generate_summary_json(
                        directory_handler.dirname,
                        run_config=merged_config,
                        strategy_number=strat_idx,
                    )
                    generate_snapshot_index(
                        directory_handler.dirname,
                        run_config=merged_config,
                        strategy_number=strat_idx,
                    )
                except Exception as e:
                    logging.warning(f"Failed to generate snapshot index: {e}")

            executed_simulations.append(mock_sim)

        # Generate inter-strategy plots if multiple strategies
        if len(executed_simulations) > 1:
            new_plot_handler.show_inter_strategy_plots(
                executed_simulations, directory_handler
            )

        # Generate main dashboard
        generate_main_dashboard(directory_handler.dirname)

        return True, output_dir, []

    except Exception as e:
        errors.append(f"Error running mock simulation: {e}")
        errors.append(traceback.format_exc())
        return False, None, errors


def verify_outputs(output_dir: Path) -> tuple[bool, list[str], dict]:
    """Verify expected outputs were created and return file counts.

    Returns:
        Tuple of (success, errors, counts) where counts has keys:
        - plots: number of PDF plot files
        - csvs: number of CSV files
        - snapshots: number of attack snapshot files
        - html: whether index.html exists
    """
    errors = []
    counts = {"plots": 0, "csvs": 0, "snapshots": 0, "html": False}

    if not output_dir.exists():
        errors.append(f"Output directory not found: {output_dir}")
        return False, errors, counts

    # Check for CSV directory and validate files
    csv_dir = output_dir / "csv"
    if not csv_dir.exists():
        errors.append("Missing csv/ directory")
    else:
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            errors.append("No CSV files found")
        else:
            counts["csvs"] = len(csv_files)
            # Validate CSV structure (has round column and data rows)
            for csv_file in csv_files:
                csv_errors = _validate_csv_file(csv_file)
                errors.extend(csv_errors)

    # Check for plots
    plots = list(output_dir.glob("*.pdf"))
    if not plots:
        errors.append("No plot files (*.pdf) found")
    else:
        counts["plots"] = len(plots)

    # Check for index.html
    if not (output_dir / "index.html").exists():
        errors.append("Missing index.html")
    else:
        counts["html"] = True

    # Count attack snapshots
    counts["snapshots"] = _count_attack_snapshots(output_dir)

    return len(errors) == 0, errors, counts


def _validate_csv_file(csv_path: Path) -> list[str]:
    """Validate a CSV file has expected structure."""
    errors = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if headers is None:
                errors.append(f"{csv_path.name}: Empty CSV file")
                return errors

            # Check for 'round' column
            if not csv_path.name.startswith("exec_stats") and "round" not in headers:
                errors.append(f"{csv_path.name}: Missing 'round' column")

            # Check for at least one data row
            first_row = next(reader, None)
            if first_row is None:
                errors.append(f"{csv_path.name}: No data rows")

    except Exception as e:
        errors.append(f"{csv_path.name}: Error reading CSV - {e}")

    return errors


def _count_attack_snapshots(output_dir: Path) -> int:
    """Count attack snapshot files across all strategies."""
    total = 0
    # Look for attack_snapshots_* directories
    for snapshots_dir in output_dir.glob("attack_snapshots_*"):
        if snapshots_dir.is_dir():
            # Count pickle or metadata JSON files
            pickles = list(snapshots_dir.glob("client_*/round_*/*.pickle"))
            jsons = list(snapshots_dir.glob("client_*/round_*/*_metadata.json"))
            total += len(pickles) if pickles else len(jsons)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Run mock simulations using recorded baseline data"
    )
    parser.add_argument(
        "--config",
        help="Single config filename to run",
    )
    parser.add_argument(
        "--all-fast",
        action="store_true",
        help="Run all fast configs",
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
        "--verify",
        action="store_true",
        help="Verify outputs after generation",
    )

    args = parser.parse_args()

    # Determine configs to run
    if args.all_fast:
        configs = FAST_CONFIGS
    elif args.config:
        configs = [args.config]
    else:
        configs = FAST_CONFIGS  # Default to all fast

    config_dir = project_root / "config" / "simulation_strategies" / args.config_dir
    baselines_dir = project_root / args.baselines_dir
    output_base = project_root / "out"

    # Check baselines exist
    missing_baselines = []
    for config_name in configs:
        baseline_name = config_name.replace(".json", ".baseline.json")
        if not (baselines_dir / baseline_name).exists():
            missing_baselines.append(config_name)

    if missing_baselines:
        console.print(f"[yellow]Missing baselines for: {missing_baselines}[/yellow]")
        console.print("[yellow]Run record_baselines.py first to create them.[/yellow]")

    # Filter to configs with baselines
    configs = [c for c in configs if c not in missing_baselines]

    if not configs:
        console.print("[red]No configs with baselines to run[/red]")
        sys.exit(1)

    console.print(f"[cyan]Running {len(configs)} mock simulation(s)...[/cyan]\n")

    passed = 0
    failed = 0

    for idx, config_name in enumerate(configs, start=1):
        console.print(f"[{idx}/{len(configs)}] {config_name}...", end=" ")

        success, output_dir, errors = run_mock_simulation(
            config_name, config_dir, baselines_dir, output_base
        )

        if success:
            if args.verify and output_dir:
                verify_ok, verify_errors, counts = verify_outputs(output_dir)
                if verify_ok:
                    console.print(
                        f"[green]OK[/green] "
                        f"({counts['plots']} plots, {counts['csvs']} CSVs, {counts['snapshots']} snapshots)"
                    )
                    passed += 1
                else:
                    console.print("[red]VERIFY FAILED[/red]")
                    for err in verify_errors:
                        console.print(f"  - {err}")
                    failed += 1
            else:
                console.print("[green]OK[/green]")
                passed += 1
        else:
            console.print("[red]FAILED[/red]")
            for err in errors:
                console.print(f"  - {err}")
            failed += 1

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
