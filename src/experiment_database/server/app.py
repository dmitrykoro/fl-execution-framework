import logging

from aiohttp import web
from sqlalchemy import text
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import ExecutedConfig, ExperimentEvent
from src.experiment_database.persistence import fill_all_tables
from src.experiment_database.session import get_db
from src.utils.common import obtain_env_var
from src.experiment_database.server.middleware import auth_middleware


async def post_experiment(request: web.Request) -> web.Response:
    """Persist a completed FL experiment run and all associated metrics."""

    try:
        data = await request.json()

    except Exception as e:  # noqa
        return web.json_response({"error": "Invalid JSON"}, status=400)

    experiment_event = None

    with get_db() as session:
        try:
            strategy_history = SimulationStrategyHistory.from_dict(data["strategy_history"])
            total_exec_time = data.get("total_exec_time")
            full_author_name = data.get("full_author_name")
            author_email = data.get("author_email")
            created_at = data.get("created_at")
            
            if created_at:
                from datetime import datetime, timezone
                created_at = datetime.fromisoformat(created_at)
            
            fill_all_tables(session, strategy_history, total_exec_time, full_author_name, author_email, created_at=created_at)

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

async def post_experiment_import(request: web.Request) -> web.Response:
    """
    Import a pre-parsed experiment from CSV/JSON files.

    Expects a flat payload (as produced by upload.py) rather than
    the SimulationStrategyHistory format used by the live FL runner.
    """
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    with get_db() as session:
        try:
            cfg_data = data["executed_config"]
            created_at_raw = data.get("created_at")
            created_at = (
                datetime.fromisoformat(created_at_raw)
                if created_at_raw
                else datetime.now(timezone.utc)
            )

            # ExecutedConfig
            executed_config = ExecutedConfig(
                id=str(uuid.uuid4()),
                json=cfg_data.get("json"),
                aggregation_strategy_keyword=cfg_data["aggregation_strategy_keyword"],
                dataset_keyword=cfg_data["dataset_keyword"],
                num_rounds=str(cfg_data["num_rounds"]),
                num_clients=str(cfg_data["num_clients"]),
                num_malicious_clients=str(cfg_data["num_malicious_clients"]),
            )
            session.add(executed_config)
            session.flush()

            # ExperimentEvent
            experiment_event = ExperimentEvent(
                created_at=created_at,
                full_author_name=data.get("full_author_name"),
                author_email=data.get("author_email"),
                is_successful=data.get("is_successful", True),
                executed_config_id=executed_config.id,
                hardware_id=None,
                total_exec_time=None,
            )
            session.add(experiment_event)
            session.flush()

            # ExperimentClients
            client_number_to_id = {}
            for client_data in data.get("clients", []):
                client = ExperimentClient(
                    id=str(uuid.uuid4()),
                    client_number=client_data["client_number"],
                    was_ever_malicious=client_data.get("was_ever_malicious", False),
                    experiment_event_id=experiment_event.id,
                )
                session.add(client)
                session.flush()
                client_number_to_id[client_data["client_number"]] = client.id

            # ClientAtRound
            for car in data.get("clients_at_round", []):
                client_id = client_number_to_id.get(car["client_number"])
                session.add(
                    ClientAtRound(
                        experiment_event_id=experiment_event.id,
                        experiment_client_id=client_id,
                        round_number=car["round_number"],
                        client_is_aggregated=car.get("is_aggregated", True),
                        attack_type=car.get("attack_type"),
                    )
                )

            # MetricInfo + MetricEntry
            for entry in data.get("metric_entries", []):
                metric_name = entry["metric_name"]
                metric_type = entry["metric_type"]

                metric_info = (
                    session.query(MetricInfo)
                    .filter_by(name=metric_name, type=metric_type)
                    .first()
                )
                if not metric_info:
                    metric_info = MetricInfo(name=metric_name, type=metric_type)
                    session.add(metric_info)
                    session.flush()

                client_number = entry.get("client_number")
                client_id = (
                    client_number_to_id.get(client_number)
                    if client_number is not None
                    else None
                )

                session.add(
                    MetricEntry(
                        id=str(uuid.uuid4()),
                        metric_info_id=metric_info.id,
                        experiment_event_id=experiment_event.id,
                        client_id=client_id,
                        value=entry["value"],
                        round_number=entry.get("round_number"),
                    )
                )

            session.commit()
            return web.json_response(
                {"status": "ok", "experiment_event_id": experiment_event.id},
                status=201,
            )

        except Exception as e:
            logging.exception("Failed to import experiment")
            return web.json_response({"error": str(e)}, status=500)


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
                        "full_author_name": exp.author.full_author_name,
                        "author_email": exp.author.author_email,
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

        finally:
            session.close()


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


async def homepage(request: web.Request) -> web.Response:
    """Homepage with summary + DB size + last executed experiment."""

    with get_db() as session:
        try:
            experiment_count = session.query(ExperimentEvent).count()

            db_bytes = session.execute(
                text("SELECT pg_database_size(current_database())")
            ).scalar()
            db_size_gb = round(db_bytes / (1024 ** 3), 3) if db_bytes is not None else "unknown"

            last_exp = (
                session.query(ExperimentEvent)
                .order_by(ExperimentEvent.id.desc())
                .first()
            )

            if last_exp:
                cfg = last_exp.executed_config  # relationship
                last_dataset = cfg.dataset_keyword if cfg else None
                author = last_exp.author
                full_author_name = author.full_author_name
                last_author = f"{full_author_name[:3]}*** ***{full_author_name[-3:]}"

                dt = last_exp.created_at
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                dt = dt.astimezone(ZoneInfo("America/New_York"))

                last_created_at = dt.strftime("%B %-d, %Y %-I:%M%p %Z")
                last_strategy = cfg.aggregation_strategy_keyword if cfg else None

            else:
                last_author = "n/a"
                last_dataset = "n/a"
                last_created_at = "n/a"
                last_strategy = "n/a"

        except Exception as e:
            experiment_count = f"INTERNAL DATABASE ERROR: {e}"
            db_size_gb = "unknown"
            last_author = "unknown"
            last_dataset = "unknown"
            last_created_at = "unknown"
            last_strategy = "unknown"

    html = f"""
    <html>
        <head>
            <title>InteFL Home</title>
            <style>
                body {{
                    margin: 0;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    font-family: Arial, sans-serif;
                    background-color: #ffffff;
                    text-align: center;
                }}
                .title {{
                    font-size: 48px;
                    margin-bottom: 24px;
                }}
                .info {{
                    font-size: 28px;
                    color: #444;
                    margin: 6px 0;
                }}
                .section {{
                    margin-top: 38px;
                    padding-top: 18px;
                    border-top: 1px solid #ddd;
                    width: min(900px, 92vw);
                }}
                .section-title {{
                    font-size: 32px;
                    margin-bottom: 12px;
                }}
                .kv {{
                    font-size: 26px;
                    color: #333;
                    margin: 6px 0;
                }}
            </style>
        </head>
        <body>
            <div class="title">InteFL Experiment Database Server</div>
            <div class="info">Stored experiments: {experiment_count}</div>
            <div class="info">Database size: {db_size_gb} GB</div>

            <div class="section">
                <div class="section-title">Last executed experiment</div>
                <div class="kv">Author: {last_author}</div>
                <div class="kv">Dataset: {last_dataset or "unknown"}</div>
                <div class="kv">Strategy: {last_strategy or "unknown"}</div>
                <div class="kv">Executed at: {last_created_at}</div>
            </div>
        </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")


app = web.Application(middlewares=[auth_middleware])
app.router.add_post("/experiments", post_experiment)
app.router.add_get("/experiments", get_experiments)
app.router.add_get("/health", health)
app.router.add_get("/get_summary", get_summary)
app.router.add_get("/", homepage)


if __name__ == "__main__":
    port = int(obtain_env_var("EXPERIMENT_SERVER_PORT"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Starting experiment server on port {port}")
    web.run_app(app, port=port)
