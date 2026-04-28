#!/usr/bin/env python3
"""Visual pipeline smoke test for the FL execution framework.

Runs the mock simulation over 5 representative configs covering the distinct
code paths in mock_simulation_runner. The CI workflow uploads the resulting
`out/` tree as a downloadable artifact bundle that reviewers open and visually
verify the plots look correct.
"""

import sys
from pathlib import Path

from rich.console import Console

from shell_scripts.mock_simulation_runner import run_mock_simulation, verify_outputs

project_root = Path(__file__).parent.parent
console = Console()

CONFIGS = [
    "femnist_krum_baseline.json",
    "femnist_krum_vs_labelflip20.json",
    "femnist_bulyan_vs_gaussnoise.json",
    "femnist_mkrum_multi_concurrent.json",
    "femnist_pid_baseline.json",
]


def run(configs: list[str], config_dir: Path | None = None, baselines_dir: Path | None = None) -> int:
    if config_dir is None:
        config_dir = project_root / "config" / "simulation_strategies" / "cpu"
    if baselines_dir is None:
        baselines_dir = project_root / "tests" / "fixtures" / "baselines"

    console.print(f"[cyan]Running {len(configs)} mock simulation(s)...[/cyan]\n")

    passed = 0
    failed = 0

    for idx, config in enumerate(configs, start=1):
        console.print(f"[{idx}/{len(configs)}] {config}...", end=" ")

        success, output_dir, errors = run_mock_simulation(
            config, config_dir, baselines_dir, project_root / "out"
        )

        if not success or output_dir is None:
            console.print(f"[red]FAILED: {errors[0] if errors else 'Unknown error'}[/red]")
            failed += 1
            continue

        verify_ok, verify_errors, counts = verify_outputs(output_dir)
        if verify_ok:
            console.print(
                f"[green]OK[/green] [dim](plots: {counts['plots']}, "
                f"csvs: {counts['csvs']}, snapshots: {counts['snapshots']})[/dim]"
            )
            passed += 1
        else:
            console.print(f"[red]VERIFY FAILED: {verify_errors[0]}[/red]")
            failed += 1

    console.print(f"\n[bold]Summary: {passed} passed, {failed} failed[/bold]")
    if failed == 0:
        console.print(
            "[dim]Download the artifact bundle from the workflow run and "
            "visually verify the plots.[/dim]"
        )
    return 0 if failed == 0 else 1


def main():
    configs = [sys.argv[1]] if len(sys.argv) > 1 else CONFIGS
    sys.exit(run(configs))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
