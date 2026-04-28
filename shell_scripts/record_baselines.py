#!/usr/bin/env python3
"""Record baseline results from running simulation configs.

This script runs simulation configs and captures their metrics
to create lightweight baseline JSON files for CI validation.

Usage:
  python record_baselines.py [config_name]
If no config_name is provided, it will run all configs defined in ALL_CONFIGS.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from shell_scripts.constants import BASELINE_FORMAT_VERSION, ALL_CONFIGS
from shell_scripts.runner.executor import ExperimentExecutor

project_root = Path(__file__).parent.parent
console = Console()


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_baseline_from_output(output_dir: Path, config_name: str, duration: float, num_clients: int) -> dict:
    """Extract baseline data from simulation output directory."""
    baseline = {
        "config": config_name,
        "recorded_at": datetime.now().isoformat(),
        "baseline_format_version": BASELINE_FORMAT_VERSION,
        "success": True,
        "num_clients": num_clients,
        "strategies": [],
        "runtime_seconds": round(duration, 1),
    }

    csv_dir = output_dir / "csv"
    if csv_dir.exists():
        for idx, csv_path in enumerate(sorted(csv_dir.glob("round_metrics_*.csv"))):
            metrics = {
                "strategy_index": idx,
                "total_rounds": 0,
                "per_round": {"aggregated_loss": [], "average_accuracy": []}
            }
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    metrics["total_rounds"] = len(rows)
                    metrics["final_accuracy"] = _safe_float(rows[-1].get("average_accuracy_history"))
                    metrics["final_loss"] = _safe_float(rows[-1].get("aggregated_loss_history"))
                    for row in rows:
                        metrics["per_round"]["aggregated_loss"].append(_safe_float(row.get("aggregated_loss_history")))
                        metrics["per_round"]["average_accuracy"].append(_safe_float(row.get("average_accuracy_history")))

            per_client_csv = csv_dir / f"per_client_metrics_{idx}.csv"
            if per_client_csv.exists():
                metrics["per_client"] = {
                    str(i): {"accuracy": [], "loss": [], "removal_criterion": [], "absolute_distance": [], "participation": []}
                    for i in range(num_clients)
                }
                with open(per_client_csv, newline="") as f:
                    for row in csv.DictReader(f):
                        for cid in range(num_clients):
                            prefix = f"client_{cid}_"
                            metrics["per_client"][str(cid)]["accuracy"].append(_safe_float(row.get(f"{prefix}accuracy_history")))
                            metrics["per_client"][str(cid)]["loss"].append(_safe_float(row.get(f"{prefix}loss_history")))
                            metrics["per_client"][str(cid)]["removal_criterion"].append(_safe_float(row.get(f"{prefix}removal_criterion_history")))
                            metrics["per_client"][str(cid)]["absolute_distance"].append(_safe_float(row.get(f"{prefix}absolute_distance_history")))
                            metrics["per_client"][str(cid)]["participation"].append(int(_safe_float(row.get(f"{prefix}aggregation_participation_history"), 1)))

            baseline["strategies"].append(metrics)

    return baseline


def main():
    configs = [sys.argv[1]] if len(sys.argv) > 1 else ALL_CONFIGS
    baselines_dir = project_root / "tests" / "fixtures" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]Recording baselines for {len(configs)} config(s)...[/cyan]\n")

    results = []
    with ExperimentExecutor(project_root=project_root) as executor:
        for idx, config_name in enumerate(configs, start=1):
            console.print(f"\n[bold cyan]=== [{idx}/{len(configs)}] {config_name} ===[/bold cyan]")

            num_clients = 10
            try:
                with open(project_root / "config" / "simulation_strategies" / "cpu" / config_name) as f:
                    num_clients = json.load(f).get("shared_settings", {}).get("num_of_clients", 10)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

            result = executor.run_experiment(config_name, idx, len(configs))

            if result.result.value == "success" and result.output_dir:
                output_path = project_root / result.output_dir
                baseline = extract_baseline_from_output(output_path, config_name, result.duration, num_clients)

                if baseline["strategies"]:
                    baseline_path = baselines_dir / config_name.replace(".json", ".baseline.json")
                    with open(baseline_path, "w") as f:
                        json.dump(baseline, f, indent=2)
                    console.print(f"[green][OK] Saved baseline: {baseline_path.name}[/green]")
                    results.append((config_name, "success", f"Saved {baseline_path.name}"))
                else:
                    console.print("[yellow][!] No strategy data found in output[/yellow]")
                    results.append((config_name, "warning", "No strategy data found"))
            else:
                console.print(f"[red][X] Failed to record baseline (Status: {result.result.value})[/red]")
                results.append((config_name, "fail", f"Status: {result.result.value}"))

            executor.cleanup(is_last=(idx == len(configs)))

    # Final Summary
    console.print("\n[bold cyan]=== Recording Summary ===[/bold cyan]")
    for config_name, status, msg in results:
        color = "green" if status == "success" else "yellow" if status == "warning" else "red"
        icon = "OK" if status == "success" else "!" if status == "warning" else "X"
        console.print(f"[{color}][{icon}] {config_name:<40} : {msg}[/{color}]")

    success_count = sum(1 for _, status, _ in results if status == "success")
    console.print(f"\n[bold]Total: {success_count}/{len(configs)} recorded successfully.[/bold]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
