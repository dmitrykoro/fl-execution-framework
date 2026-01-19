import os

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
