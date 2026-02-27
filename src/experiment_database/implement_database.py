import asyncio
import logging
import os
import sqlalchemy
import aiohttp

from dotenv import load_dotenv

from src.experiment_database.persistence import (
    create_clients,
    create_executed_config,
    create_experiment_event,
    get_or_create_hardware,
    persist_client_round_metrics,
    persist_clients_at_round,
    persist_round_metrics,
)
from src.experiment_database.session import LocalDatabaseSession, NeonDatabaseSession
from src.federated_simulation import FederatedSimulation
from src.utils.common import obtain_env_var

load_dotenv(".env.private")

EXPERIMENT_SERVER_URL = os.environ.get("EXPERIMENT_SERVER_URL")


def database_logger(strategy: FederatedSimulation) -> None:
    """Write experiment results to Neon and/or the local experiment server."""
    if NeonDatabaseSession is not None:
        _write_to_database(strategy, NeonDatabaseSession)

    if EXPERIMENT_SERVER_URL:
        asyncio.run(_send_to_server(strategy, EXPERIMENT_SERVER_URL))


async def _send_to_server(strategy: FederatedSimulation, server_url: str) -> None:
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
        "username": obtain_env_var("FL_USERNAME"),
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/experiments",
                json=payload,
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


def _write_to_database(strategy: FederatedSimulation, SessionFactory) -> None:
    """Persist experiment results directly to the database via SQLAlchemy."""
    session = SessionFactory()
    experiment_event = None

    try:
        hardware = get_or_create_hardware(
            session,
            cpu=strategy.strategy_history.host_cpu,
            arch=strategy.strategy_history.training_device_arch,
            ram=strategy.strategy_history.host_ram,
            gpu_name=strategy.strategy_history.training_device_name,
            gpu_ram=strategy.strategy_history.training_device_ram,
            is_executed_on=strategy.strategy_history.training_device_type,
        )
        executed_config = create_executed_config(
            session=session,
            strategy_config=strategy.strategy_config,
            num_of_malicious_clients=strategy.strategy_history.total_malicious_clients,
        )
        experiment_event = create_experiment_event(
            session=session,
            executed_config=executed_config,
            hardware=hardware,
            username=obtain_env_var("FL_USERNAME"),
        )
        persist_round_metrics(
            session=session,
            experiment_event=experiment_event,
            rounds_info=strategy.strategy_history.rounds_history,
        )
        _, experiment_client_id_map = create_clients(
            session=session,
            experiment_event=experiment_event,
            num_clients=strategy.strategy_config.num_of_clients,
            malicious_ids=strategy.strategy_history.get_malicious_client_ids(),
        )
        persist_clients_at_round(
            session=session,
            experiment_event=experiment_event,
            clients_info=strategy.strategy_history.get_all_clients(),
            experiment_client_id_map=experiment_client_id_map,
        )
        persist_client_round_metrics(
            session=session,
            experiment_event=experiment_event,
            clients_info=strategy.strategy_history.get_all_clients(),
            experiment_client_id_map=experiment_client_id_map,
        )
        experiment_event.is_successful = True
        experiment_event.total_exec_time = strategy.get_time()
        session.commit()

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