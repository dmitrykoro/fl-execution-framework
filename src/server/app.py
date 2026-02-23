import logging
import os
from aiohttp import web

from src.experiment_database.session import LocalDatabaseSession
from src.experiment_database.persistence import (
    create_executed_config,
    get_or_create_hardware,
    create_experiment_event,
    create_clients,
    persist_clients_at_round,
    persist_round_metrics,
    persist_client_round_metrics,
)
from src.data_models.simulation_strategy_history import SimulationStrategyHistory


async def post_experiment(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    session = LocalDatabaseSession()
    experiment_event = None

    try:
        history = SimulationStrategyHistory.from_dict(data["strategy_history"])
        total_exec_time = data.get("total_exec_time")

        hardware = get_or_create_hardware(
            session,
            cpu=data.get("cpu", "unknown"),
            arch=data.get("arch", "unknown"),
            ram=data.get("ram", "unknown"),
        )
        executed_config = create_executed_config(
            session=session,
            strategy_config=history.strategy_config,
            num_of_malicious_clients=history.total_malicious_clients,
        )
        experiment_event = create_experiment_event(
            session=session,
            executed_config=executed_config,
            hardware=hardware,
        )
        persist_round_metrics(
            session=session,
            experiment_event=experiment_event,
            rounds_info=history.rounds_history,
        )
        clients, experiment_client_id_map = create_clients(
            session=session,
            experiment_event=experiment_event,
            num_clients=history.strategy_config.num_of_clients,
            malicious_ids=history.get_malicious_client_id(),
        )
        persist_clients_at_round(
            session=session,
            experiment_event=experiment_event,
            clients_info=history.get_all_clients(),
            experiment_client_id_map=experiment_client_id_map,
        )
        persist_client_round_metrics(
            session=session,
            experiment_event=experiment_event,
            clients_info=history.get_all_clients(),
            experiment_client_id_map=experiment_client_id_map,
        )
        experiment_event.is_successful = True
        experiment_event.total_exec_time = total_exec_time
        session.commit()

        return web.json_response({"status": "ok"}, status=201)

    except Exception as e:
        logging.exception("Failed to persist experiment")
        if experiment_event is not None:
            try:
                experiment_event.error_message = str(e)
                experiment_event.is_successful = False
                session.add(experiment_event)
                session.commit()
            except Exception:
                pass
        return web.json_response({"error": str(e)}, status=500)

    finally:
        session.close()


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy"})


app = web.Application()
app.router.add_post("/experiments", post_experiment)
app.router.add_get("/health", health)

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8080))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Starting experiment server on port {port}")
    web.run_app(app, port=port)