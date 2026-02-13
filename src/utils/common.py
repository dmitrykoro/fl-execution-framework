import os
import json

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
        return var_value

    var_value = input(f"Please enter {var_key}: ").strip()
    private_path.touch(exist_ok=True)

    with private_path.open("a") as f:
        f.write(f"\n{var_key}={var_value}\n")

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


