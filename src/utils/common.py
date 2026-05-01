import os
import json
import logging
import asyncio

from dotenv import dotenv_values
from pathlib import Path


def obtain_env_var(var_key: str):
    """
    Check if .env value exists; if not, prompt user to enter through shell.
    """

    shared_path = Path(".env.public")
    private_path = Path(".env.private")

    full_env_config = {
        **(dotenv_values(shared_path) if shared_path.exists() else {}),
        **(dotenv_values(private_path) if private_path.exists() else {}),
        ** os.environ,
    }

    var_value = full_env_config.get(var_key)
    if var_value not in (None, ""):
        os.environ[var_key] = var_value
        return var_value

    var_value = input(f"Please enter {var_key}: ").strip()
    private_path.touch(exist_ok=True)

    with private_path.open("a") as f:
        f.write(f"{var_key}={var_value}\n")

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


