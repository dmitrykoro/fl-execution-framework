import logging
import shutil
import subprocess
import sqlalchemy
import aiohttp
import requests
from pathlib import Path

from src.experiment_database.persistence import fill_all_tables
from src.experiment_database.remote_access import (
    RemoteServerUnavailable,
    initialize_remote_access,
    remote_credentials,
)
from src.experiment_database.session import get_db, local_db_path, resolve_db_mode, set_db_mode
from src.federated_simulation import FederatedSimulation
from src.utils.common import obtain_env_var, run_async_from_sync

DEFAULT_AUTHOR_NAME = "Anonymous"
DEFAULT_AUTHOR_EMAIL = "anonymous@localhost"


def _git_config_value(key: str) -> str | None:
    """Read a value from the local git configuration, if git is available."""

    if shutil.which("git") is None:
        return None

    try:
        result = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    value = result.stdout.strip()

    return value or None


class DatabaseHandler:

    def __init__(self):
        self.db_mode = resolve_db_mode()

        self.full_author_name = obtain_env_var(
            "FULL_AUTHOR_NAME",
            default=_git_config_value("user.name") or DEFAULT_AUTHOR_NAME,
        )
        self.author_email = obtain_env_var(
            "AUTHOR_EMAIL",
            default=_git_config_value("user.email") or DEFAULT_AUTHOR_EMAIL,
        )

        if self.db_mode == "server":
            # Prompts for anything missing when a terminal is available, so a new group
            # member is asked for their token on the first run and never silently
            # uploads nothing.
            initialize_remote_access()

            try:
                self.EXPERIMENT_SERVER_URL, self.EXPERIMENT_SERVER_TOKEN = remote_credentials()

            except RemoteServerUnavailable:
                self._fall_back_to_local_database(
                    "no experiment server URL and token were provided"
                )

            else:
                if not self._ensure_token_valid():
                    self._fall_back_to_local_database(
                        "the experiment server is unreachable or rejected the token"
                    )

    def _fall_back_to_local_database(self, reason: str) -> None:
        """
        Switch to the local SQLite database instead of failing the run.

        Uploading to the research group's experiment server needs credentials that
        are only handed out inside the group. Without them the simulation still runs
        and its results are still persisted, just locally.
        """

        self.db_mode = "local"
        set_db_mode("local")

        logging.warning(
            f"Experiment results will be stored locally because {reason}. "
            f"Database file: {local_db_path().resolve()}. "
            "Set EXPERIMENT_SERVER_URL and EXPERIMENT_SERVER_TOKEN in .env.private "
            "to upload results to the experiment server instead."
        )

    @staticmethod
    def _build_local_dashboard_url(simulation_strategy: FederatedSimulation) -> str | None:
        directory_handler = getattr(simulation_strategy, "directory_handler", None)
        output_dir = getattr(directory_handler, "dirname", None) if directory_handler else None

        if not output_dir:
            return None

        return (Path(output_dir).resolve() / "index.html").as_uri()

    def _ensure_token_valid(self) -> bool:
        """Try to fetch server status from /health with the provided token. Fail if token is invalid."""

        url = f"{self.EXPERIMENT_SERVER_URL}/health"

        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=10)

            if resp.status_code in (401, 403):
                logging.error(f"Forbidden when requesting {url}. Token rejected.")
                return False

            if resp.status_code != 200:
                logging.error(f"Health check failed ({resp.status_code}): {resp.text}")
                return False

            try:
                payload = resp.json()
            except Exception:
                payload = resp.text

            logging.info(f"Experiment server health OK: {payload}")
            return True

        except requests.RequestException as e:
            logging.error(f"Health check request failed: {e}")
            return False


    def _auth_headers(self) -> dict:
        """Common auth headers for experiment server requests."""
        return {
            "Authorization": f"Bearer {self.EXPERIMENT_SERVER_TOKEN}",
            "Accept": "application/json",
        }

    def update_database(self, simulation_strategy: FederatedSimulation) -> None:
        """Update database with experiment results"""

        if self.db_mode in ("local", "direct", "neon"):
            self._direct_write_to_db(simulation_strategy)

        elif self.db_mode == "server":
            run_async_from_sync(self._server_write_to_db(simulation_strategy))

    async def _server_write_to_db(self, strategy: FederatedSimulation) -> None:
        """Send experiment results to the remote experiment server via HTTP POST."""

        local_dashboard_url = self._build_local_dashboard_url(strategy)
        payload = {
            "strategy_history": strategy.strategy_history.to_dict(),
            "total_exec_time": strategy.get_time(),
            "cpu": strategy.strategy_history.host_cpu,
            "arch": strategy.strategy_history.training_device_arch,
            "ram": strategy.strategy_history.host_ram,
            "gpu_name": strategy.strategy_history.training_device_name,
            "gpu_ram": strategy.strategy_history.training_device_ram,
            "is_executed_on": strategy.strategy_history.training_device_type,
            "local_dashboard_url": local_dashboard_url,
            "full_author_name": self.full_author_name,
            "author_email": self.author_email
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.EXPERIMENT_SERVER_URL}/experiments",
                        json=payload,
                        headers=self._auth_headers(),
                        timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 201:
                        logging.error(
                            f"Server rejected experiment data ({resp.status}): "
                            f"{await resp.text()}"
                        )
                    else:
                        logging.info("Experiment data sent to remote server successfully")

        except Exception as e:
            logging.error(f"Failed to send experiment data to remote server: {e}")


    def _direct_write_to_db(self, strategy: FederatedSimulation) -> None:
        """Persist experiment results directly to the database via SQLAlchemy."""

        experiment_event = None

        with get_db() as session:
            try:
                strategy_history = strategy.strategy_history
                total_exec_time = strategy.get_time()
                local_dashboard_url = self._build_local_dashboard_url(strategy)

                fill_all_tables(
                    session,
                    strategy_history,
                    total_exec_time,
                    self.full_author_name,
                    self.author_email,
                    local_dashboard_url=local_dashboard_url,
                )

                if self.db_mode == "local":
                    logging.info(
                        f"Experiment results stored in {local_db_path().resolve()}"
                    )

            except sqlalchemy.exc.OperationalError as e:
                logging.error(
                    f"Error while trying to insert experiment metrics to the database: {e}"
                )

            except Exception as e:
                if experiment_event is not None:
                    try:
                        experiment_event.error_message = str(e)
                        experiment_event.is_successful = False
                        experiment_event.total_exec_time = strategy.get_time()
                        session.add(experiment_event)
                        session.commit()
                    except Exception:
                        pass

                # A failed local write must not discard a finished simulation: the CSVs
                # and plots in out/ are already on disk.
                if self.db_mode == "local":
                    logging.error(
                        f"Failed to store experiment results in the local database: {e}"
                    )
                    return

                raise

            finally:
                session.close()
