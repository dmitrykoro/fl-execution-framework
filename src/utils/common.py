import os
import json

from dotenv import load_dotenv
from pathlib import Path


def obtain_env_var(var_key: str):
    """
    Check if .env value exists; if not, prompt user to enter through shell.
    """
    env_path = Path(".env")

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        var_value = os.getenv(var_key)

        if var_value:
            return var_value

    new_value = input(f"Please enter {var_key}: ").strip()
    with open(env_path, "a") as f:
        f.write(f"\n{var_key}={new_value}\n")

    return new_value


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


