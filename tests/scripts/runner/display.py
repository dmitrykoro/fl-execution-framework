"""Display formatting utilities for experiment runner UI."""

from rich.console import Console
from typing import Optional, List, Dict, Any

console = Console()


class ExperimentDisplay:
    """Format and display experiment runner output."""

    @staticmethod
    def show_header(
        config_name: str, title: Optional[str], index: int, total: int
    ) -> None:
        """Display experiment header.

        Args:
            config_name: Config filename
            title: Human-readable title (optional)
            index: Current experiment index
            total: Total number of experiments
        """
        if title:
            console.print(f'\n[cyan][{index}/{total}] Running "{title}"[/cyan]')
            console.print(f"[dim]  ({config_name})[/dim]")
        else:
            console.print(f"\n[cyan][{index}/{total}] Running {config_name}...[/cyan]")

    @staticmethod
    def show_success(index: int, total: int, duration: float) -> None:
        """Display success message.

        Args:
            index: Current experiment index
            total: Total number of experiments
            duration: Execution duration in seconds
        """
        console.print(
            f"[green][OK] Completed {index}/{total}[/green] ({duration:.1f}s)"
        )

    @staticmethod
    def show_failure(config_name: str, exit_code: int, duration: float) -> None:
        """Display failure message.

        Args:
            config_name: Config filename
            exit_code: Process exit code
            duration: Execution duration in seconds
        """
        console.print(
            f"[red][FAIL] Failed on {config_name}[/red] "
            f"(exit code: {exit_code}, {duration:.1f}s)"
        )

    @staticmethod
    def show_timeout(config_name: str, timeout_seconds: int) -> None:
        """Display timeout message.

        Args:
            config_name: Config filename
            timeout_seconds: Timeout value in seconds
        """
        if timeout_seconds >= 60:
            timeout_str = f"{timeout_seconds // 60} minutes"
        else:
            timeout_str = f"{timeout_seconds} seconds"

        console.print(
            f"[yellow][TIMEOUT] Timed out on {config_name}[/yellow] "
            f"(exceeded {timeout_str})"
        )

    @staticmethod
    def show_cleanup(mode: str) -> None:
        """Display cleanup message.

        Args:
            mode: Cleanup mode (basic, aggressive)
        """
        console.print(f"[dim]Running cleanup (mode: {mode})...[/dim]")

    @staticmethod
    def show_cleanup_complete() -> None:
        """Display cleanup complete message."""
        console.print("[dim]Garbage collection complete[/dim]")

    @staticmethod
    def show_cleanup_pause(seconds: int) -> None:
        """Display cleanup pause message.

        Args:
            seconds: Pause duration in seconds
        """
        console.print(f"[dim]Pausing {seconds} seconds for resource cleanup...[/dim]")

    @staticmethod
    def show_timing_stats(
        device: str, count: int, avg: float, min_time: float, max_time: float
    ) -> None:
        """Display timing database statistics.

        Args:
            device: Device type (cuda, cpu, auto)
            count: Number of recorded timings
            avg: Average duration
            min_time: Minimum duration
            max_time: Maximum duration
        """
        console.print(
            f"[dim]Timing DB ({device.upper()}): {count} configs, "
            f"avg {avg}s (range: {min_time}-{max_time}s)[/dim]"
        )


class SummaryDisplay:
    """Display execution summary reports."""

    @staticmethod
    def show_summary(
        results: Dict[str, List[str]], skipped: Optional[List[str]] = None
    ) -> None:
        """Display batch execution summary.

        Args:
            results: Results dictionary with completed/failed/timedout lists
            skipped: List of skipped configs (optional)
        """
        console.print("\n" + "=" * 50)
        console.print("[bold cyan]Execution Summary[/bold cyan]")
        console.print("=" * 50 + "\n")

        total_errors = len(results["failed"]) + len(results["timedout"])
        configs_run = results["total"]

        if skipped:
            console.print(
                f"[dim]Skipped {len(skipped)} already-completed config(s)[/dim]\n"
            )

        if total_errors == 0:
            if configs_run == 0:
                console.print(
                    "[green]All selected configs were already completed![/green]"
                )
            else:
                console.print(
                    f"[green]All {configs_run} config(s) completed successfully![/green]"
                )
        else:
            console.print(
                f"[red]{total_errors}/{configs_run} config(s) failed:[/red]\n"
            )

            if results["timedout"]:
                console.print(
                    f"[yellow]Timed out ({len(results['timedout'])}):[/yellow]"
                )
                for config in results["timedout"]:
                    console.print(f"  [yellow][T] {config}[/yellow]")
                console.print()

            if results["failed"]:
                console.print(f"[red]Failed ({len(results['failed'])}):[/red]")
                for config in results["failed"]:
                    console.print(f"  [red][X] {config}[/red]")
                console.print()

        console.print("=" * 50)


# Convenience functions for backwards compatibility
def display_summary(
    results: Dict[str, Any], skipped: Optional[List[str]] = None
) -> None:
    """Convenience function for displaying summary."""
    SummaryDisplay.show_summary(results, skipped)
