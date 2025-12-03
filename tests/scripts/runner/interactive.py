"""Interactive prompts for experiment runner."""

from typing import List, Optional
from InquirerPy import inquirer  # type: ignore[import-untyped]
from InquirerPy.base.control import Choice  # type: ignore[import-untyped]
from rich.console import Console
from rich.table import Table

console = Console()


def prompt_resume(completed_configs: List[str]) -> bool:
    """Prompt user to resume from previous run.

    Args:
        completed_configs: List of already completed configs

    Returns:
        True to resume, False to start fresh
    """
    console.print("\n[yellow]Found previous run state[/yellow]")
    console.print(f"Completed configs from previous run: {len(completed_configs)}")

    # Show completed configs in a table
    table = Table(show_header=False, box=None)
    for config in completed_configs[:10]:  # Show first 10
        table.add_row("[OK]", config)
    if len(completed_configs) > 10:
        table.add_row("...", f"and {len(completed_configs) - 10} more")

    console.print(table)
    console.print()

    return inquirer.confirm(  # type: ignore[attr-defined]
        message="Resume from last run?", default=True
    ).execute()


def prompt_background_mode() -> bool:
    """Prompt for background/foreground execution mode.

    Returns:
        True for background mode, False for foreground
    """
    console.print("\n[bold cyan]Run Mode Selection[/bold cyan]")
    console.print("Background mode allows the script to continue running")
    console.print("even if you close the terminal (useful for long runs).\n")

    choice = inquirer.select(  # type: ignore[attr-defined]
        message="Select run mode:",
        choices=[
            Choice(value=False, name="Foreground - Stay attached to terminal"),
            Choice(
                value=True, name="Background - Detach and continue if terminal closes"
            ),
        ],
        default=False,
    ).execute()

    return choice


def prompt_log_level() -> str:
    """Prompt for log level selection.

    Returns:
        Selected log level (DEBUG, INFO, WARNING, ERROR)
    """
    console.print("\n[bold cyan]Log Level Selection[/bold cyan]\n")

    return inquirer.select(  # type: ignore[attr-defined]
        message="Select log level:",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    ).execute()


def prompt_timeout() -> Optional[int]:
    """Prompt for timeout selection.

    Returns:
        Timeout in seconds, or None for unlimited
    """
    console.print("\n[bold cyan]Timeout Selection[/bold cyan]\n")

    choice = inquirer.select(  # type: ignore[attr-defined]
        message="Select timeout per config:",
        choices=[
            Choice(value=None, name="No timeout (unlimited)"),
            Choice(value=1800, name="30 minutes"),
            Choice(value=3600, name="1 hour"),
            Choice(value=7200, name="2 hours"),
            Choice(value=14400, name="4 hours"),
            Choice(value="custom", name="Custom timeout"),
        ],
        default=3600,
    ).execute()

    if choice == "custom":
        seconds = inquirer.number(  # type: ignore[attr-defined]
            message="Enter timeout in seconds:", min_allowed=1, validate=lambda x: x > 0
        ).execute()
        return int(seconds)

    return choice


def prompt_cleanup_mode() -> str:
    """Prompt for cleanup mode selection.

    Returns:
        Cleanup mode (none, basic, aggressive)
    """
    console.print("\n[bold cyan]Resource Cleanup Selection[/bold cyan]")
    console.print("This helps prevent hangs on Windows by forcing garbage")
    console.print("collection and clearing temp files between runs.\n")

    choice = inquirer.select(  # type: ignore[attr-defined]
        message="Select cleanup mode:",
        choices=[
            Choice(
                value="none", name="No cleanup (faster, but may hang on long batches)"
            ),
            Choice(value="basic", name="Basic cleanup (force Python GC between runs)"),
            Choice(
                value="aggressive",
                name="Aggressive cleanup (GC + clear temp files + pause) - RECOMMENDED for Windows",
            ),
        ],
        default="aggressive",
    ).execute()

    return choice


def prompt_config_selection(
    all_configs: List[str], completed_configs: Optional[List[str]] = None
) -> List[str]:
    """Prompt for config file selection.

    Args:
        all_configs: All available config files
        completed_configs: Already completed configs (will be marked)

    Returns:
        List of selected config filenames
    """
    console.print("\n[bold cyan]Config Selection[/bold cyan]\n")

    completed_set = set(completed_configs or [])

    # Create choices with indicators for completed configs
    choices = []
    for config in all_configs:
        if config in completed_set:
            name = f"[OK] {config} [dim](already completed)[/dim]"
        else:
            name = config
        choices.append(
            Choice(value=config, name=name, enabled=(config not in completed_set))
        )

    selected = inquirer.checkbox(  # type: ignore[attr-defined]
        message="Select configs to run (space to toggle, enter to confirm):",
        choices=choices,
        instruction="(Use arrow keys to navigate, space to select/deselect, 'a' to toggle all)",
        validate=lambda result: len(result) > 0,
        invalid_message="Please select at least one config",
        transformer=lambda result: f"{len(result)} config(s) selected",
    ).execute()

    return selected


def display_settings_summary(
    run_mode: str,
    log_level: str,
    timeout: Optional[int],
    cleanup_mode: str,
    selected_configs: List[str],
    skipped_configs: List[str],
) -> None:
    """Display summary of selected settings.

    Args:
        run_mode: "Foreground" or "Background"
        log_level: Selected log level
        timeout: Timeout in seconds or None
        cleanup_mode: Selected cleanup mode
        selected_configs: Configs to run
        skipped_configs: Configs that will be skipped
    """
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]Run Settings Summary[/bold cyan]")
    console.print("=" * 50)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[bold]Run Mode:[/bold]", run_mode)
    table.add_row("[bold]Log Level:[/bold]", log_level)

    if timeout:
        timeout_str = f"{timeout} seconds ({timeout // 60} minutes)"
    else:
        timeout_str = "No timeout (unlimited)"
    table.add_row("[bold]Timeout:[/bold]", timeout_str)

    table.add_row("[bold]Cleanup Mode:[/bold]", cleanup_mode.capitalize())
    table.add_row("[bold]Configs to run:[/bold]", str(len(selected_configs)))

    if skipped_configs:
        table.add_row(
            "[bold]Configs to skip:[/bold]",
            f"{len(skipped_configs)} (already completed)",
        )

    console.print(table)
    console.print("=" * 50 + "\n")
