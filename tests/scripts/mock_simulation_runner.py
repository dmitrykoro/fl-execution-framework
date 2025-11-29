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
import json
import sys
from pathlib import Path
from rich.console import Console

# Project root for path resolution
project_root = Path(__file__).parent.parent.parent
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

    # Import here to avoid loading heavy dependencies unless needed
    from src.data_models.simulation_strategy_config import StrategyConfig
    from src.data_models.simulation_strategy_history import SimulationStrategyHistory
    from src.output_handlers import new_plot_handler
    from src.output_handlers.directory_handler import DirectoryHandler
    from src.attack_utils.snapshot_html_reports import generate_main_dashboard

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
        import traceback

        errors.append(f"Error running mock simulation: {e}")
        errors.append(traceback.format_exc())
        return False, None, errors


def verify_outputs(output_dir: Path) -> tuple[bool, list[str]]:
    """Verify expected outputs were created."""
    errors = []

    if not output_dir.exists():
        errors.append(f"Output directory not found: {output_dir}")
        return False, errors

    # Check for CSV directory
    csv_dir = output_dir / "csv"
    if not csv_dir.exists():
        errors.append("Missing csv/ directory")
    else:
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            errors.append("No CSV files found")

    # Check for plots
    plots = list(output_dir.glob("*.pdf"))
    if not plots:
        errors.append("No plot files (*.pdf) found")

    # Check for index.html
    if not (output_dir / "index.html").exists():
        errors.append("Missing index.html")

    return len(errors) == 0, errors


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
