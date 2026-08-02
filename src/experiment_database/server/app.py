import logging
import math
import re
import decimal
import uuid

from aiohttp import web
from sqlalchemy import text
from datetime import datetime, date
from zoneinfo import ZoneInfo

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import (
    ClientAtRound,
    ConfigurationChatMessage,
    ConfigurationSession,
    DatabaseAnalysisChatMessage,
    DatabaseAnalysisSession,
    ExecutedConfig,
    ExperimentAuthor,
    ExperimentClient,
    ExperimentEvent,
    MetricEntry,
    MetricInfo,
)
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
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|merge|comment|refresh|vacuum|analyze|set)\b",
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


def _normalize_author_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned.upper() if cleaned else None


def _resolve_author(session, full_author_name: str | None, author_email: str | None) -> ExperimentAuthor | None:
    normalized_email = _normalize_author_value(author_email)
    normalized_name = _normalize_author_value(full_author_name)

    if not normalized_email:
        return None

    author = (
        session.query(ExperimentAuthor)
        .filter(ExperimentAuthor.author_email == normalized_email)
        .first()
    )
    if author is None:
        author = ExperimentAuthor(
            full_author_name=normalized_name or normalized_email,
            author_email=normalized_email,
        )
        session.add(author)
        session.flush()
        return author

    if normalized_name and author.full_author_name != normalized_name:
        author.full_author_name = normalized_name
        session.add(author)
        session.flush()

    return author


def _serialize_clients_at_round(session, experiment_event_id: int, limit: int = 20) -> list[dict]:
    rows = (
        session.query(ClientAtRound, ExperimentClient)
        .join(ExperimentClient, ClientAtRound.experiment_client_id == ExperimentClient.id)
        .filter(ClientAtRound.experiment_event_id == experiment_event_id)
        .order_by(ClientAtRound.round_number.asc(), ExperimentClient.client_number.asc())
        .limit(limit)
        .all()
    )

    return [
        {
            "round_number": client_at_round.round_number,
            "client_number": client.client_number if client else None,
            "is_aggregated": client_at_round.client_is_aggregated,
            "attack_type": client_at_round.attack_type,
        }
        for client_at_round, client in rows
    ]


def _serialize_metric_entries(session, experiment_event_id: int, limit: int = 200) -> list[dict]:
    def _query_metric_rows(per_round_only: bool, row_limit: int | None = None):
        query = (
            session.query(MetricEntry, MetricInfo, ExperimentClient)
            .join(MetricInfo, MetricEntry.metric_info_id == MetricInfo.id)
            .outerjoin(ExperimentClient, MetricEntry.client_id == ExperimentClient.id)
            .filter(MetricEntry.experiment_event_id == experiment_event_id)
        )

        if per_round_only:
            query = query.filter(MetricEntry.client_id.is_(None))
        else:
            query = query.filter(MetricEntry.client_id.is_not(None))

        query = query.order_by(
            MetricEntry.round_number.asc(),
            MetricInfo.name.asc(),
            ExperimentClient.client_number.asc(),
        )

        if row_limit is not None:
            query = query.limit(row_limit)

        return query.all()

    def _serialize(rows) -> list[dict]:
        return [
            {
                "round_number": metric_entry.round_number,
                "metric_name": metric_info.name if metric_info else None,
                "metric_type": metric_info.type if metric_info else None,
                "value": metric_entry.value,
                "client_number": client.client_number if client else None,
            }
            for metric_entry, metric_info, client in rows
        ]

    per_round_rows = _query_metric_rows(per_round_only=True)
    serialized = _serialize(per_round_rows)

    remaining = limit - len(serialized)
    if remaining <= 0:
        return serialized[:limit]

    per_client_rows = _query_metric_rows(per_round_only=False, row_limit=remaining)
    serialized.extend(_serialize(per_client_rows))
    return serialized


def _serialize_analysis_session(session_row: DatabaseAnalysisSession) -> dict:
    return {
        "id": session_row.id,
        "created_at": session_row.created_at.isoformat() if session_row.created_at else None,
        "payload": session_row.payload or "",
        "author_email": session_row.author.author_email if session_row.author else None,
        "author_name": session_row.author.full_author_name if session_row.author else None,
        "messages": [
            {
                "id": message.id,
                "role": message.creator_role,
                "payload": message.payload,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in sorted(session_row.chat_messages, key=lambda item: ((item.created_at.isoformat() if item.created_at else ""), item.id))
            if not message.is_removed
        ],
    }


def _serialize_configuration_session(session_row: ConfigurationSession) -> dict:
    return {
        "id": session_row.id,
        "created_at": session_row.created_at.isoformat() if session_row.created_at else None,
        "author_email": session_row.author.author_email if session_row.author else None,
        "author_name": session_row.author.full_author_name if session_row.author else None,
        "messages": [
            {
                "id": message.id,
                "role": message.creator_role,
                "payload": message.payload,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in sorted(session_row.chat_messages, key=lambda item: ((item.created_at.isoformat() if item.created_at else ""), item.id))
            if not message.is_removed
        ],
    }


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
            local_dashboard_url = data.get("local_dashboard_url")

            fill_all_tables(
                session,
                strategy_history,
                total_exec_time,
                full_author_name,
                author_email,
                local_dashboard_url=local_dashboard_url,
            )

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
                        "local_dashboard_url": exp.local_dashboard_url,
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
                        "clients_at_round": _serialize_clients_at_round(session, exp.id),
                        "metric_entries": _serialize_metric_entries(session, exp.id),
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


async def create_database_analysis_session(request: web.Request) -> web.Response:
    payload = await request.json()

    with get_db() as session:
        try:
            author = _resolve_author(
                session,
                full_author_name=payload.get("author_name"),
                author_email=payload.get("author_email"),
            )
            analysis_session = DatabaseAnalysisSession(
                id=str(payload.get("id") or uuid.uuid4()),
                author=author,
                payload=str(payload.get("payload", "") or ""),
            )
            session.add(analysis_session)
            session.commit()
            session.refresh(analysis_session)
            return web.json_response(_serialize_analysis_session(analysis_session), status=201)
        except Exception as e:
            logging.exception("Failed to create database analysis session")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def list_database_analysis_sessions(request: web.Request) -> web.Response:
    author_email = _normalize_author_value(request.query.get("author_email"))

    with get_db() as session:
        try:
            query = session.query(DatabaseAnalysisSession).filter(DatabaseAnalysisSession.is_removed.is_(False))
            if author_email:
                query = query.join(DatabaseAnalysisSession.author).filter(ExperimentAuthor.author_email == author_email)

            rows = query.order_by(DatabaseAnalysisSession.created_at.desc()).all()
            items = [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "payload_preview": (row.payload or "")[:140],
                    "message_count": sum(1 for msg in row.chat_messages if not msg.is_removed),
                }
                for row in rows
            ]
            return web.json_response({"items": items})
        except Exception as e:
            logging.exception("Failed to list database analysis sessions")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def get_database_analysis_session(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]

    with get_db() as session:
        try:
            row = (
                session.query(DatabaseAnalysisSession)
                .filter(
                    DatabaseAnalysisSession.id == session_id,
                    DatabaseAnalysisSession.is_removed.is_(False),
                )
                .first()
            )
            if row is None:
                return web.json_response({"error": "Analysis session not found"}, status=404)
            return web.json_response(_serialize_analysis_session(row))
        except Exception as e:
            logging.exception("Failed to load database analysis session")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def append_database_analysis_message(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    payload = await request.json()
    role = str(payload.get("role", "")).strip().lower()
    message_text = str(payload.get("payload", "")).strip()

    if role not in {"user", "agent"}:
        return web.json_response({"error": "role must be user or agent"}, status=400)
    if not message_text:
        return web.json_response({"error": "payload must not be empty"}, status=400)

    with get_db() as session:
        try:
            row = (
                session.query(DatabaseAnalysisSession)
                .filter(
                    DatabaseAnalysisSession.id == session_id,
                    DatabaseAnalysisSession.is_removed.is_(False),
                )
                .first()
            )
            if row is None:
                return web.json_response({"error": "Analysis session not found"}, status=404)

            message = DatabaseAnalysisChatMessage(
                database_analysis_session_id=row.id,
                creator_role=role,
                payload=message_text,
            )
            session.add(message)
            session.commit()
            return web.json_response({"ok": True, "id": message.id}, status=201)
        except Exception as e:
            logging.exception("Failed to append database analysis chat message")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def update_database_analysis_payload(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    payload = await request.json()
    markdown = str(payload.get("payload", "") or "")
    mode = str(payload.get("mode", "replace")).strip().lower()

    if mode not in {"replace", "append"}:
        return web.json_response({"error": "mode must be replace or append"}, status=400)

    with get_db() as session:
        try:
            row = (
                session.query(DatabaseAnalysisSession)
                .filter(
                    DatabaseAnalysisSession.id == session_id,
                    DatabaseAnalysisSession.is_removed.is_(False),
                )
                .first()
            )
            if row is None:
                return web.json_response({"error": "Analysis session not found"}, status=404)

            if mode == "append" and row.payload:
                row.payload = f"{row.payload.rstrip()}\n\n{markdown.lstrip()}".strip()
            else:
                row.payload = markdown

            session.add(row)
            session.commit()
            session.refresh(row)
            return web.json_response({"ok": True, "payload": row.payload or ""})
        except Exception as e:
            logging.exception("Failed to update database analysis payload")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def create_configuration_session(request: web.Request) -> web.Response:
    payload = await request.json()

    with get_db() as session:
        try:
            author = _resolve_author(
                session,
                full_author_name=payload.get("author_name"),
                author_email=payload.get("author_email"),
            )
            configuration_session = ConfigurationSession(
                id=str(payload.get("id") or uuid.uuid4()),
                author=author,
            )
            session.add(configuration_session)
            session.commit()
            session.refresh(configuration_session)
            return web.json_response(_serialize_configuration_session(configuration_session), status=201)
        except Exception as e:
            logging.exception("Failed to create configuration session")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def get_configuration_session(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]

    with get_db() as session:
        try:
            row = (
                session.query(ConfigurationSession)
                .filter(
                    ConfigurationSession.id == session_id,
                    ConfigurationSession.is_removed.is_(False),
                )
                .first()
            )
            if row is None:
                return web.json_response({"error": "Configuration session not found"}, status=404)
            return web.json_response(_serialize_configuration_session(row))
        except Exception as e:
            logging.exception("Failed to load configuration session")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def list_configuration_sessions(request: web.Request) -> web.Response:
    author_email = _normalize_author_value(request.query.get("author_email"))

    with get_db() as session:
        try:
            query = session.query(ConfigurationSession).filter(ConfigurationSession.is_removed.is_(False))
            if author_email:
                query = query.join(ConfigurationSession.author).filter(ExperimentAuthor.author_email == author_email)

            rows = query.order_by(ConfigurationSession.created_at.desc()).all()
            items = [
                {
                    "id": row.id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "message_count": sum(1 for msg in row.chat_messages if not msg.is_removed),
                }
                for row in rows
            ]
            return web.json_response({"items": items})
        except Exception as e:
            logging.exception("Failed to list configuration sessions")
            return web.json_response({"error": str(e)}, status=500)
        finally:
            session.close()


async def append_configuration_message(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    payload = await request.json()
    role = str(payload.get("role", "")).strip().lower()
    message_text = str(payload.get("payload", "")).strip()

    if role not in {"user", "agent"}:
        return web.json_response({"error": "role must be user or agent"}, status=400)
    if not message_text:
        return web.json_response({"error": "payload must not be empty"}, status=400)

    with get_db() as session:
        try:
            row = (
                session.query(ConfigurationSession)
                .filter(
                    ConfigurationSession.id == session_id,
                    ConfigurationSession.is_removed.is_(False),
                )
                .first()
            )
            if row is None:
                return web.json_response({"error": "Configuration session not found"}, status=404)

            message = ConfigurationChatMessage(
                configuration_session_id=row.id,
                creator_role=role,
                payload=message_text,
            )
            session.add(message)
            session.commit()
            return web.json_response({"ok": True, "id": message.id}, status=201)
        except Exception as e:
            logging.exception("Failed to append configuration chat message")
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

                if not author:  # a person did not fill out name and email when running the framework
                    last_author = "Anonymous"
                else:
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
app.router.add_post("/analysis_sessions", create_database_analysis_session)
app.router.add_get("/analysis_sessions", list_database_analysis_sessions)
app.router.add_get("/analysis_sessions/{session_id}", get_database_analysis_session)
app.router.add_post("/analysis_sessions/{session_id}/messages", append_database_analysis_message)
app.router.add_put("/analysis_sessions/{session_id}/payload", update_database_analysis_payload)
app.router.add_post("/configuration_sessions", create_configuration_session)
app.router.add_get("/configuration_sessions", list_configuration_sessions)
app.router.add_get("/configuration_sessions/{session_id}", get_configuration_session)
app.router.add_post("/configuration_sessions/{session_id}/messages", append_configuration_message)
app.router.add_get("/health", health)
app.router.add_get("/get_summary", get_summary)
app.router.add_get("/", homepage)


if __name__ == "__main__":
    port = int(obtain_env_var("EXPERIMENT_SERVER_PORT"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Starting experiment server on port {port}")
    web.run_app(app, port=port)
