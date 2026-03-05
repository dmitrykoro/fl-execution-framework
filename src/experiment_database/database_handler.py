import logging
import sqlalchemy
import aiohttp
import requests

from src.experiment_database.persistence import fill_all_tables
from src.experiment_database.session import get_db
from src.federated_simulation import FederatedSimulation
from src.utils.common import obtain_env_var, run_async_from_sync


EXPERIMENT_DB_MODE = obtain_env_var("EXPERIMENT_DB_MODE")
FULL_AUTHOR_NAME = obtain_env_var("FULL_AUTHOR_NAME")
AUTHOR_EMAIL = obtain_env_var("AUTHOR_EMAIL")


class DatabaseHandler:

    def __init__(self):
        if EXPERIMENT_DB_MODE in ("direct", "neon"):
            pass

        elif EXPERIMENT_DB_MODE == "server":
            self.EXPERIMENT_SERVER_URL = obtain_env_var("EXPERIMENT_SERVER_URL")
            # self.EXPERIMENT_SERVER_PORT = obtain_env_var("EXPERIMENT_SERVER_PORT")
            self.EXPERIMENT_SERVER_TOKEN = obtain_env_var("EXPERIMENT_SERVER_TOKEN")

            if not self._ensure_token_valid():
                raise Exception("Experiment server token is invalid or server not running.")

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

        if EXPERIMENT_DB_MODE in ("direct", "neon"):
            self._direct_write_to_db(simulation_strategy)

        elif EXPERIMENT_DB_MODE == "server":
            run_async_from_sync(self._server_write_to_db(simulation_strategy))

    async def _server_write_to_db(self, strategy: FederatedSimulation) -> None:
        """Send experiment results to the remote experiment server via HTTP POST."""

        payload = {
            "strategy_history": strategy.strategy_history.to_dict(),
            "total_exec_time": strategy.get_time(),
            "cpu": strategy.strategy_history.host_cpu,
            "arch": strategy.strategy_history.training_device_arch,
            "ram": strategy.strategy_history.host_ram,
            "gpu_name": strategy.strategy_history.training_device_name,
            "gpu_ram": strategy.strategy_history.training_device_ram,
            "is_executed_on": strategy.strategy_history.training_device_type,
            "full_author_name": FULL_AUTHOR_NAME,
            "author_email": AUTHOR_EMAIL
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

                fill_all_tables(session, strategy_history, total_exec_time, FULL_AUTHOR_NAME, AUTHOR_EMAIL)

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
                raise

            finally:
                session.close()
