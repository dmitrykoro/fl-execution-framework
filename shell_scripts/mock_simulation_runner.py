#!/usr/bin/env python3
"""Mock simulation runner with real strategy execution for CI testing.

Runs federated learning simulations using baseline data instead of actual
training, while executing real strategy aggregation logic. Validates that
the full output generation pipeline (plots, CSVs, snapshots, HTML) works
correctly without the overhead of distributed execution.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import traceback
from pathlib import Path

import torch
from rich.console import Console

from src.output_handlers.attack_snapshot_handler import save_attack_snapshot
from src.attack_utils.snapshot_html_reports import (
    generate_main_dashboard,
    generate_snapshot_index,
    generate_summary_json,
)
from src.config_loaders.validate_strategy_config import _populate_client_selection
from src.data_models.simulation_strategy_config import StrategyConfig
from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.output_handlers import new_plot_handler
from src.output_handlers.directory_handler import DirectoryHandler
from shell_scripts.constants import ALL_CONFIGS

project_root = Path(__file__).parent.parent
console = Console()


def load_baseline(config_name: str, baselines_dir: Path) -> dict | None:
    """Load pre-recorded baseline data for a simulation config."""
    baseline_name = config_name.replace(".json", ".baseline.json")
    baseline_path = baselines_dir / baseline_name

    if not baseline_path.exists():
        return None

    with open(baseline_path) as f:
        return json.load(f)


def load_config(config_name: str, config_dir: Path) -> dict:
    """Load simulation configuration from JSON file."""
    config_path = config_dir / config_name
    with open(config_path) as f:
        return json.load(f)


def populate_history_from_baseline(
    strategy_history: SimulationStrategyHistory,
    baseline_strategy: dict,
    num_clients: int,
) -> None:
    """Populate strategy history from pre-recorded baseline data."""
    per_round = baseline_strategy.get("per_round", {})
    per_client = baseline_strategy.get("per_client", {})
    baseline_rounds = baseline_strategy.get("total_rounds", 10)
    config_rounds = strategy_history.strategy_config.num_of_rounds
    total_rounds = min(baseline_rounds, config_rounds)

    for client_id in range(num_clients):
        client_data = per_client.get(str(client_id), {})
        for round_num in range(1, total_rounds + 1):
            idx = round_num - 1

            def get_val(key):
                data_list = client_data.get(key, [0.0] * baseline_rounds)
                return data_list[idx] if idx < len(data_list) else 0.0

            strategy_history.insert_single_client_history_entry(
                client_id=client_id,
                current_round=round_num,
                removal_criterion=get_val("removal_criterion"),
                absolute_distance=get_val("absolute_distance"),
                loss=get_val("loss"),
                accuracy=get_val("accuracy"),
                aggregation_participation=int(get_val("participation")),
            )

    aggregated_loss = per_round.get("aggregated_loss", [0.0] * baseline_rounds)
    for round_num in range(1, total_rounds + 1):
        idx = round_num - 1
        strategy_history.insert_round_history_entry(
            score_calculation_time_nanos=0,
            removal_threshold=0.0,
            loss_aggregated=aggregated_loss[idx] if idx < len(aggregated_loss) else 0.0,
        )


class MockFederatedSimulation:
    """Lightweight simulation wrapper for testing output handlers."""

    def __init__(
        self,
        strategy_config: StrategyConfig,
        strategy_history: SimulationStrategyHistory,
        attack_schedule: list | None = None,
    ) -> None:
        self.strategy_config = strategy_config
        self.strategy_history = strategy_history
        self._attack_schedule = attack_schedule or []

    def get_attack_schedule_as_dict(self) -> list:
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
    """Generate mock attack snapshots for CI testing."""
    snapshots_generated = 0

    for attack in attack_schedule:
        attack_type = attack.get("attack_type", "unknown")
        start_round = attack.get("start_round", 1)
        end_round = attack.get("end_round", total_rounds)

        selection = attack.get("selection_strategy", "percentage")
        if selection == "percentage":
            percentage = attack.get("malicious_percentage", 0.2)
            num_malicious = max(1, int(num_clients * percentage))
            malicious_clients = list(range(num_malicious))
        elif selection == "specific":
            malicious_clients = attack.get("client_ids", [0])
        else:
            malicious_clients = [0]

        for round_num in range(start_round, min(end_round, total_rounds) + 1):
            for client_id in malicious_clients:
                mock_data = torch.rand(max_samples, 1, 28, 28)
                mock_labels = torch.randint(0, 10, (max_samples,))

                original_labels = (mock_labels + 1) % 10 if attack_type == "label_flipping" else mock_labels.clone()

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
                    logging.warning(f"Failed to generate mock snapshot for client {client_id}, round {round_num}: {e}")

    return snapshots_generated


class MockDatasetHandler:
    def __init__(self, malicious_clients, poisoned_client_ids):
        self.malicious_clients = malicious_clients
        self.poisoned_client_ids = poisoned_client_ids


def run_mock_simulation(
    config_name: str,
    config_dir: Path,
    baselines_dir: Path,
    output_base: Path,
) -> tuple[bool, Path | None, list[str]]:
    """Run mock simulation using baseline data."""
    errors = []

    baseline = load_baseline(config_name, baselines_dir)
    if not baseline:
        errors.append(f"Baseline not found for {config_name}. Run record_baselines.py first.")
        return False, None, errors

    try:
        config = load_config(config_name, config_dir)
    except Exception as e:
        errors.append(f"Failed to load config: {e}")
        return False, None, errors

    try:
        directory_handler = DirectoryHandler(custom_dirname=config_name.replace(".json", ""))
        assert directory_handler.dirname is not None
        dirname: str = directory_handler.dirname
        output_dir = Path(dirname)

        shared_settings = config.get("shared_settings", {})
        strategies = config.get("simulation_strategies", [{}])
        num_clients = baseline.get("num_clients", shared_settings.get("num_of_clients", 10))

        executed_simulations = []

        for strat_idx, strategy_overrides in enumerate(strategies):
            merged_config = {**shared_settings, **strategy_overrides}
            merged_config["strategy_number"] = strat_idx

            _populate_client_selection(merged_config)

            strategy_config = StrategyConfig.from_dict(merged_config)
            strategy_config.strategy_number = strat_idx

            malicious_clients = set()
            poisoned_client_ids = set()
            for attack in merged_config.get("attack_schedule", []):
                selected = attack.get("malicious_client_ids", [])
                malicious_clients.update(selected)
                poisoned_client_ids.update(selected)

            dataset_handler = MockDatasetHandler(
                malicious_clients=malicious_clients,
                poisoned_client_ids=poisoned_client_ids
            )

            strategy_history = SimulationStrategyHistory(
                strategy_config=strategy_config,
                dataset_handler=dataset_handler,  # type: ignore[arg-type]
            )

            if strat_idx < len(baseline.get("strategies", [])):
                baseline_strategy = baseline["strategies"][strat_idx]
                populate_history_from_baseline(strategy_history, baseline_strategy, num_clients)
            else:
                raise ValueError(f"Strategy {strat_idx} not found in baseline for {config_name}")

            strategy_history.calculate_additional_rounds_data()

            mock_sim = MockFederatedSimulation(
                strategy_config=strategy_config,
                strategy_history=strategy_history,
                attack_schedule=merged_config.get("attack_schedule", []),
            )

            if strategy_config.save_plots:
                new_plot_handler.show_plots_within_strategy(
                    mock_sim,  # type: ignore[arg-type]
                    directory_handler,
                )

            if strategy_config.save_csv:
                directory_handler.save_csv_and_config(strategy_history)

            attack_schedule = merged_config.get("attack_schedule", [])
            save_snapshots = merged_config.get("save_attack_snapshots", False)
            if attack_schedule and str(save_snapshots).strip().lower() == "true":
                total_rounds = merged_config.get("num_of_rounds", 10)
                snapshot_format = merged_config.get("attack_snapshot_format", "pickle")
                max_samples = merged_config.get("snapshot_max_samples", 5)

                generate_mock_attack_snapshots(
                    attack_schedule=attack_schedule,
                    output_dir=dirname,
                    num_clients=num_clients,
                    total_rounds=total_rounds,
                    strategy_number=strat_idx,
                    max_samples=max_samples,
                    save_format=snapshot_format,
                )

                try:
                    generate_summary_json(dirname, run_config=merged_config, strategy_number=strat_idx)
                    generate_snapshot_index(dirname, run_config=merged_config, strategy_number=strat_idx)
                except Exception as e:
                    logging.warning(f"Failed to generate snapshot index: {e}")

            executed_simulations.append(mock_sim)

        if len(executed_simulations) > 1:
            new_plot_handler.show_inter_strategy_plots(executed_simulations, directory_handler)

        generate_main_dashboard(dirname)

        return True, output_dir, []

    except Exception as e:
        errors.append(f"Error running mock simulation: {e}")
        errors.append(traceback.format_exc())
        return False, None, errors


def verify_outputs(output_dir: Path) -> tuple[bool, list[str], dict]:
    """Verify expected output artifacts were created."""
    errors = []
    counts = {"plots": 0, "csvs": 0, "snapshots": 0, "html": False}

    if not output_dir.exists():
        errors.append(f"Output directory not found: {output_dir}")
        return False, errors, counts

    csv_dir = output_dir / "csv"
    if not csv_dir.exists():
        errors.append("Missing csv/ directory")
    else:
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            errors.append("No CSV files found")
        else:
            counts["csvs"] = len(csv_files)
            for csv_file in csv_files:
                csv_errors = _validate_csv_file(csv_file)
                errors.extend(csv_errors)

    plots = list(output_dir.glob("*.pdf"))
    if not plots:
        errors.append("No plot files (*.pdf) found")
    else:
        counts["plots"] = len(plots)

    if not (output_dir / "index.html").exists():
        errors.append("Missing index.html")
    else:
        counts["html"] = True

    counts["snapshots"] = _count_attack_snapshots(output_dir)

    return len(errors) == 0, errors, counts


def _validate_csv_file(csv_path: Path) -> list[str]:
    """Validate CSV has header row and at least one data row."""
    errors = []
    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if headers is None:
                errors.append(f"{csv_path.name}: Empty CSV file")
                return errors

            if not csv_path.name.startswith("exec_stats") and "round" not in headers:
                errors.append(f"{csv_path.name}: Missing 'round' column")

            first_row = next(reader, None)
            if first_row is None:
                errors.append(f"{csv_path.name}: No data rows")

    except Exception as e:
        errors.append(f"{csv_path.name}: Error reading CSV - {e}")

    return errors


def _count_attack_snapshots(output_dir: Path) -> int:
    """Count attack snapshot files across all strategy directories."""
    total = 0
    for snapshots_dir in output_dir.glob("attack_snapshots_*"):
        if snapshots_dir.is_dir():
            pickles = list(snapshots_dir.glob("client_*/round_*/*.pickle"))
            jsons = list(snapshots_dir.glob("client_*/round_*/*_metadata.json"))
            total += len(pickles) if pickles else len(jsons)
    return total


def main():
    configs = [sys.argv[1]] if len(sys.argv) > 1 else ALL_CONFIGS
    
    config_dir = project_root / "config" / "simulation_strategies" / "cpu"
    baselines_dir = project_root / "tests" / "fixtures" / "baselines"
    output_base = project_root / "out"

    console.print(f"[cyan]Running {len(configs)} mock simulation(s)...[/cyan]\n")

    passed = 0
    failed = 0

    for idx, config_name in enumerate(configs, start=1):
        console.print(f"[{idx}/{len(configs)}] {config_name}...", end=" ")

        success, output_dir, errors = run_mock_simulation(
            config_name, config_dir, baselines_dir, output_base
        )

        if success:
            if output_dir:
                verify_ok, verify_errors, counts = verify_outputs(output_dir)
                if verify_ok:
                    console.print(f"[green]OK[/green] [dim]({counts['plots']} plots, {counts['csvs']} CSVs, {counts['snapshots']} snapshots)[/dim]")
                    passed += 1
                else:
                    console.print(f"[red]VERIFY FAILED: {verify_errors[0]}[/red]")
                    failed += 1
            else:
                console.print("[green]OK[/green]")
                passed += 1
        else:
            console.print(f"[red]FAILED: {errors[0] if errors else 'Unknown error'}[/red]")
            failed += 1

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
