import os
import json
import logging
import asyncio
import sys

from dotenv import dotenv_values
from pathlib import Path


def is_interactive_shell() -> bool:
    """
    Report whether the process may block on stdin for user input.

    Returns False when stdin is not a terminal (subprocess, CI, nohup) or when
    INTEFL_NON_INTERACTIVE is set, so that missing configuration degrades to a
    default instead of hanging forever on an unanswerable prompt.
    """

    if os.environ.get("INTEFL_NON_INTERACTIVE", "").strip().lower() in ("1", "true", "yes"):
        return False

    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def read_env_var(var_key: str, default: str | None = None) -> str | None:
    """
    Look up a configuration value without ever prompting for it.

    Use this wherever blocking on stdin would be wrong, such as inside a request
    handler on the aiohttp event loop. Values already resolved by obtain_env_var()
    are visible here because it publishes them to os.environ.
    """

    shared_path = Path(".env.public")
    private_path = Path(".env.private")

    full_env_config = {
        **(dotenv_values(shared_path) if shared_path.exists() else {}),
        **(dotenv_values(private_path) if private_path.exists() else {}),
        ** os.environ,
    }

    return full_env_config.get(var_key) or default


def obtain_env_var(
        var_key: str,
        default: str | None = None,
        prompt_hint: str | None = None
) -> str | None:
    """
    Resolve a configuration value from .env.public, .env.private or the environment.

    If the value is absent, prompt for it through the shell and persist the answer
    to .env.private. Prompting only happens on an interactive terminal; headless
    runs (and empty answers) fall back to `default` instead, so that a checkout
    without research-group credentials still starts up.

    Args:
        var_key: Name of the variable to resolve.
        default: Value to use when nothing is configured and nothing is entered.
        prompt_hint: Extra text shown in the prompt, e.g. how to opt out.

    Returns:
        The resolved value, or `default` (which may be None) if there is none.
    """

    private_path = Path(".env.private")

    var_value = read_env_var(var_key)
    if var_value:
        return _remember_env_var(var_key, var_value)

    if not is_interactive_shell():
        logging.debug(
            f"{var_key} is not configured and no terminal is available to ask for it; "
            f"using default: {default!r}"
        )
        return _remember_env_var(var_key, default)

    hint = f" [{prompt_hint}]" if prompt_hint else ""

    try:
        var_value = input(f"Please enter {var_key}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        var_value = ""

    if not var_value:
        return _remember_env_var(var_key, default)

    private_path.touch(exist_ok=True)

    with private_path.open("a") as f:
        f.write(f"{var_key}={var_value}\n")

    return _remember_env_var(var_key, var_value)


def _remember_env_var(var_key: str, var_value: str | None) -> str | None:
    """Publish a resolved value to os.environ so later lookups stay consistent."""

    if var_value is not None:
        os.environ[var_key] = var_value

    return var_value


def load_json_as_str(filepath: str) -> str:
    """
    Load a JSON file and return its contents as a formatted string.

    Args:
        filepath: Path to JSON file.

    Returns:
        JSON content as a pretty-formatted string.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return json.dumps(data, indent=2)


def run_async_from_sync(coroutine) -> None:
    """
    Run an async coroutine from sync code.
    - If a loop is running: schedule it with create_task
    - If no loop is running: run it to completion with asyncio.run
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread -> run synchronously
        asyncio.run(coroutine)
        return

    task = loop.create_task(coroutine)

    def _log_result(t: asyncio.Task):
        try:
            t.result()
        except Exception:
            logging.exception("Async task failed")

    task.add_done_callback(_log_result)


