#!/usr/bin/env python3
"""CI smoke test: replay every smoke config through the mock runner."""

import sys

from rich.console import Console

from shell_scripts.constants import BASELINES_DIR, PROJECT_ROOT, SMOKE_CONFIGS, SMOKE_CONFIGS_DIR
from shell_scripts.mock_simulation_runner import run_mock_simulation, verify_outputs

console = Console()


def run(configs: list[str]) -> int:
    console.print(f"[cyan]Running {len(configs)} mock simulation(s)...[/cyan]\n")

    passed = 0
    failed = 0

    for idx, config in enumerate(configs, start=1):
        console.print(f"[{idx}/{len(configs)}] {config}...", end=" ")

        success, output_dir, errors = run_mock_simulation(
            config, SMOKE_CONFIGS_DIR, BASELINES_DIR, PROJECT_ROOT / "out"
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
    configs = [sys.argv[1]] if len(sys.argv) > 1 else SMOKE_CONFIGS
    sys.exit(run(configs))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
