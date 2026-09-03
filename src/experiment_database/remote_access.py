"""
Availability of the research group's remote experiment server.

Single source of truth for "can this checkout reach the group's database?", shared by
the simulation runner and by the frontend API. A checkout without credentials runs in
local mode, where results go to a local SQLite database and the features that query the
group's database are switched off rather than failing with an authorization error.
"""

import logging

from src.experiment_database.session import resolve_db_mode
from src.utils.common import obtain_env_var, read_env_var

LOCAL_MODE_MESSAGE = (
    "This feature needs the research group's experiment server, which this checkout "
    "is not connected to. Experiments are being stored in a local database instead. "
    "Set EXPERIMENT_SERVER_URL and EXPERIMENT_SERVER_TOKEN in .env.private to enable it."
)

TOKEN_PROMPT_HINT = "leave empty to work locally without the group's database"

_initialized = False


class RemoteServerUnavailable(Exception):
    """Raised when an operation needs the experiment server but no credentials are configured."""

    def __init__(self, message: str = LOCAL_MODE_MESSAGE):
        super().__init__(message)


def initialize_remote_access(prompt: bool = True) -> bool:
    """
    Resolve the experiment server credentials once, at startup.

    Long-running processes must call this before serving requests: obtain_env_var() may
    prompt on a terminal, which would block an event loop if it happened inside a request
    handler. Resolved values are published to os.environ, so simulation subprocesses
    spawned later inherit them and upload their results to the same server.

    Args:
        prompt: Whether missing values may be asked for interactively.

    Returns:
        True if the experiment server is configured.
    """

    global _initialized

    if prompt:
        obtain_env_var("EXPERIMENT_SERVER_URL")
        obtain_env_var("EXPERIMENT_SERVER_TOKEN", prompt_hint=TOKEN_PROMPT_HINT)

    _initialized = True
    available = is_remote_available()

    if available:
        logging.info(
            f"Connected to the experiment server at {read_env_var('EXPERIMENT_SERVER_URL')}. "
            "Experiment results will be uploaded to it."
        )
    else:
        logging.warning(
            "Running in local mode: no experiment server credentials are configured. "
            "Results will be stored in a local database and database-backed features "
            "are unavailable. Set EXPERIMENT_SERVER_URL and EXPERIMENT_SERVER_TOKEN "
            "in .env.private to connect to the research group's server."
        )

    return available


def remote_credentials() -> tuple[str, str]:
    """
    Return (url, token) for the experiment server.

    Raises:
        RemoteServerUnavailable: If either value is missing.
    """

    url = read_env_var("EXPERIMENT_SERVER_URL")
    token = read_env_var("EXPERIMENT_SERVER_TOKEN")

    if not url or not token:
        raise RemoteServerUnavailable()

    return url, token


def is_remote_available() -> bool:
    """Report whether experiment server credentials are configured. Never prompts."""

    return bool(read_env_var("EXPERIMENT_SERVER_URL") and read_env_var("EXPERIMENT_SERVER_TOKEN"))


def remote_auth_headers() -> dict[str, str]:
    """
    Return the authorization headers for the experiment server.

    Raises:
        RemoteServerUnavailable: If no credentials are configured.
    """

    _, token = remote_credentials()

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def remote_status() -> dict[str, object]:
    """Describe the effective storage mode for the frontend."""

    available = is_remote_available()
    configured_mode = resolve_db_mode()

    # `server` mode without credentials falls back to the local database at runtime.
    effective_mode = (
        "local" if configured_mode == "server" and not available else configured_mode
    )

    return {
        "remote_available": available,
        "db_mode": effective_mode,
        "message": None if available else LOCAL_MODE_MESSAGE,
    }
