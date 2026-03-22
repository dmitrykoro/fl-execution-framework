import logging
import math
import re
import decimal

from aiohttp import web
from sqlalchemy import text
from datetime import datetime, date
from zoneinfo import ZoneInfo

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import ExecutedConfig, ExperimentEvent
from src.experiment_database.persistence import fill_all_tables
from src.experiment_database.session import get_db
from src.utils.common import obtain_env_var
from src.experiment_database.server.middleware import auth_middleware


def _query_int(
        request: web.Request,
        name: str,
        default: int,
        minimum: int = 1,
        maximum: int | None = None,
) -> int:
    raw = request.query.get(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(text=f"{name} must be an integer")

    if value < minimum:
        raise web.HTTPBadRequest(text=f"{name} must be >= {minimum}")

    if maximum is not None and value > maximum:
        raise web.HTTPBadRequest(text=f"{name} must be <= {maximum}")

    return value


READONLY_SQL_RE = re.compile(r"^\s*(with\b.+?\bselect\b|select\b)", re.IGNORECASE | re.DOTALL)
FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|merge|comment|refresh|vacuum|analyze)\b",
    re.IGNORECASE,
)


def _validate_readonly_sql(sql: str) -> str:
    stripped = sql.strip()

    if not stripped:
        raise web.HTTPBadRequest(text="sql must not be empty")

    if ";" in stripped.rstrip(";"):
        raise web.HTTPBadRequest(text="Only a single SQL statement is allowed")

    if not READONLY_SQL_RE.match(stripped):
        raise web.HTTPBadRequest(text="Only read-only SELECT / WITH queries are allowed")

    if FORBIDDEN_SQL_RE.search(stripped):
        raise web.HTTPBadRequest(text="Data manipulation statements are not allowed")

    return stripped


def _json_safe_value(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, decimal.Decimal):
        return float(value)

    return value


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

            fill_all_tables(session, strategy_history, total_exec_time, full_author_name, author_email)

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
    """Return paginated experiment events with config, hardware, and metrics."""

    page = _query_int(request, "page", default=1, minimum=1)
    page_size = _query_int(request, "page_size", default=6, minimum=1, maximum=50)
    offset = (page - 1) * page_size

    with get_db() as session:
        try:
            total_count = session.query(ExperimentEvent).count()
            experiments = (
                session.query(ExperimentEvent)
                .order_by(ExperimentEvent.id.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )

            result = []
            for exp in experiments:
                cfg = exp.executed_config
                hw = exp.hardware
                author = exp.author
                result.append(
                    {
                        "id": exp.id,
                        "created_at": exp.created_at.isoformat() if exp.created_at else None,
                        "wandb_url": exp.wandb_url,
                        "is_successful": exp.is_successful,
                        "total_exec_time": exp.total_exec_time,
                        "error_message": exp.error_message,
                        "full_author_name": author.full_author_name if author else None,
                        "author_email": author.author_email if author else None,
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

            total_pages = math.ceil(total_count / page_size) if total_count else 0
            return web.json_response(
                {
                    "count": len(result),
                    "total": total_count,
                    "page": page,
                    "page_size": page_size,
                    "pages": total_pages,
                    "experiments": result,
                }
            )

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


async def execute_query(request: web.Request) -> web.Response:
    payload = await request.json()
    sql = _validate_readonly_sql(str(payload.get("sql", "")))

    with get_db() as session:
        try:
            result = session.execute(text(sql))
            mappings = result.mappings().all()
            columns = list(result.keys())
            rows = [{key: _json_safe_value(value) for key, value in dict(row).items()} for row in mappings]

            return web.json_response(
                {
                    "columns": columns,
                    "rows": rows,
                    "row_count": len(rows),
                }
            )

        except web.HTTPException:
            raise

        except Exception as e:
            logging.exception("Failed to execute read-only SQL query")
            return web.json_response({"error": str(e)}, status=500)

        finally:
            session.close()


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
app.router.add_post("/query", execute_query)
app.router.add_get("/health", health)
app.router.add_get("/get_summary", get_summary)
app.router.add_get("/", homepage)


if __name__ == "__main__":
    port = int(obtain_env_var("EXPERIMENT_SERVER_PORT"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Starting experiment server on port {port}")
    web.run_app(app, port=port)
