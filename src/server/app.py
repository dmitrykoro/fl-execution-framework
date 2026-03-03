import logging
import os

from aiohttp import web

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import ExecutedConfig, ExperimentEvent
from src.experiment_database.persistence import (
    create_clients,
    create_executed_config,
    create_experiment_event,
    get_or_create_hardware,
    persist_client_round_metrics,
    persist_clients_at_round,
    persist_round_metrics,
)
from src.experiment_database.session import get_db, LocalDatabaseSession


async def post_experiment(request: web.Request) -> web.Response:
    """Persist a completed FL experiment run and all associated metrics."""
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
            cpu=history.host_cpu,
            arch=history.training_device_arch,
            ram=history.host_ram,
            gpu_name=history.training_device_name,
            gpu_ram=history.training_device_ram,
            is_executed_on=history.training_device_type,
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
            username=data.get("username"),
        )
        persist_round_metrics(
            session=session,
            experiment_event=experiment_event,
            rounds_info=history.rounds_history,
        )
        _, experiment_client_id_map = create_clients(
            session=session,
            experiment_event=experiment_event,
            num_clients=history.strategy_config.num_of_clients,
            malicious_ids=history.get_malicious_client_ids(),
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


async def get_experiments(request: web.Request) -> web.Response:
    """Return the 20 most recent experiment events with config, hardware, and metrics."""
    with get_db() as session:
        try:
            experiments = (
                session.query(ExperimentEvent)
                .order_by(ExperimentEvent.id.desc())
                .limit(20)
                .all()
            )

            result = []
            for exp in experiments:
                cfg = exp.executed_config
                hw = exp.hardware
                result.append(
                    {
                        "id": exp.id,
                        "created_at": exp.created_at.isoformat() if exp.created_at else None,
                        "is_successful": exp.is_successful,
                        "total_exec_time": exp.total_exec_time,
                        "error_message": exp.error_message,
                        "username": exp.username,
                        "config": {
                            "strategy": cfg.aggregation_strategy_keyword if cfg else None,
                            "dataset": cfg.dataset_keyword if cfg else None,
                            "num_rounds": cfg.num_rounds if cfg else None,
                            "num_clients": cfg.num_clients if cfg else None,
                            "num_malicious_clients": cfg.num_malicious_clients if cfg else None,
                            "json": cfg.json if cfg else None,
                        },
                        "hardware": {
                            "cpu": hw.cpu_name if hw else None,
                            "arch": hw.cpu_arch if hw else None,
                            "ram": hw.ram_amount if hw else None,
                            "gpu_name": hw.gpu_name if hw else None,
                            "gpu_ram": hw.gpu_ram if hw else None,
                            "is_executed_on": hw.is_executed_on if hw else None,
                        },
                        "clients_at_round": [
                            {
                                "round_number": c.round_number,
                                "client_number": c.client.client_number if c.client else None,
                                "is_aggregated": c.client_is_aggregated,
                                "attack_type": c.attack_type,
                            }
                            for c in (exp.clients_at_round[:20] if exp.clients_at_round else [])
                        ],
                        "metric_entries": [
                            {
                                "round_number": m.round_number,
                                "metric_name": m.metric_info.name if m.metric_info else None,
                                "metric_type": m.metric_info.type if m.metric_info else None,
                                "value": m.value,
                                "client_number": m.client.client_number if m.client else None,
                            }
                            for m in (exp.metric_entries[:200] if exp.metric_entries else [])
                        ],
                    }
                )

            return web.json_response({"count": len(result), "experiments": result})

        except Exception as e:
            logging.exception("Failed to fetch experiments")
            return web.json_response({"error": str(e)}, status=500)



async def get_summary(request: web.Request) -> web.Response:
    """Return the total count of stored experiment configs."""
    with get_db() as session:
        try:
            count = session.query(ExecutedConfig).count()
            return web.json_response({"experiment_configs": count})
        except Exception as e:
            logging.exception("Failed to fetch summary")
            return web.json_response({"error": str(e)}, status=500)


async def health(request: web.Request) -> web.Response:
    """Return a simple health check response."""
    return web.json_response({"status": "healthy"})


app = web.Application()
app.router.add_post("/experiments", post_experiment)
app.router.add_get("/experiments", get_experiments)
app.router.add_get("/health", health)
app.router.add_get("/", get_summary)


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8080))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Starting experiment server on port {port}")
    web.run_app(app, port=port)