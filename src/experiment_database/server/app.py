import logging

from aiohttp import web
from sqlalchemy import text, func, case
from datetime import datetime
from zoneinfo import ZoneInfo

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.experiment_database.models import (
    ExecutedConfig,
    ExperimentEvent,
    MetricEntry,
    MetricInfo,
    ExperimentAuthor,
)
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
            strategy_history = SimulationStrategyHistory.from_dict(
                data["strategy_history"]
            )
            total_exec_time = data.get("total_exec_time")
            full_author_name = data.get("full_author_name")
            author_email = data.get("author_email")
            created_at = data.get("created_at")

            if created_at:
                from datetime import datetime, timezone

                created_at = datetime.fromisoformat(created_at)

            fill_all_tables(
                session,
                strategy_history,
                total_exec_time,
                full_author_name,
                author_email,
                created_at=created_at,
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
                        "created_at": exp.created_at.isoformat()
                        if exp.created_at
                        else None,
                        "is_successful": exp.is_successful,
                        "total_exec_time": exp.total_exec_time,
                        "error_message": exp.error_message,
                        "full_author_name": exp.author.full_author_name,
                        "author_email": exp.author.author_email,
                        "config": {
                            "strategy": cfg.aggregation_strategy_keyword
                            if cfg
                            else None,
                            "dataset": cfg.dataset_keyword if cfg else None,
                            "num_rounds": cfg.num_rounds if cfg else None,
                            "num_clients": cfg.num_clients if cfg else None,
                            "num_malicious_clients": cfg.num_malicious_clients
                            if cfg
                            else None,
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
                                "client_number": c.client.client_number
                                if c.client
                                else None,
                                "is_aggregated": c.client_is_aggregated,
                                "attack_type": c.attack_type,
                            }
                            for c in (
                                exp.clients_at_round[:20]
                                if exp.clients_at_round
                                else []
                            )
                        ],
                        "metric_entries": [
                            {
                                "round_number": m.round_number,
                                "metric_name": m.metric_info.name
                                if m.metric_info
                                else None,
                                "metric_type": m.metric_info.type
                                if m.metric_info
                                else None,
                                "value": m.value,
                                "client_number": m.client.client_number
                                if m.client
                                else None,
                            }
                            for m in (
                                exp.metric_entries[:200] if exp.metric_entries else []
                            )
                        ],
                    }
                )

            return web.json_response({"count": len(result), "experiments": result})

        except Exception as e:
            logging.exception("Failed to fetch experiments")
            return web.json_response({"error": str(e)}, status=500)

        finally:
            session.close()


async def get_experiment_by_id(request: web.Request) -> web.Response:
    """
    Return full detail for a single experiment by ID.
    GET /experiments/{id}
    """
    try:
        experiment_id = int(request.match_info.get("id"))
    except (ValueError, TypeError):
        return web.json_response({"error": "Invalid experiment ID"}, status=400)

    with get_db() as session:
        try:
            exp = (
                session.query(ExperimentEvent)
                .filter(ExperimentEvent.id == experiment_id)
                .first()
            )

            if exp is None:
                return web.json_response(
                    {"error": f"Experiment {experiment_id} not found"}, status=404
                )

            cfg = exp.executed_config
            hw = exp.hardware

            result = {
                "id": exp.id,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "is_successful": exp.is_successful,
                "total_exec_time": exp.total_exec_time,
                "error_message": exp.error_message,
                "full_author_name": exp.author.full_author_name if exp.author else None,
                "author_email": exp.author.author_email if exp.author else None,
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
                    for c in (exp.clients_at_round or [])
                ],
                "metric_entries": [
                    {
                        "round_number": m.round_number,
                        "metric_name": m.metric_info.name if m.metric_info else None,
                        "metric_type": m.metric_info.type if m.metric_info else None,
                        "value": m.value,
                        "client_number": m.client.client_number if m.client else None,
                    }
                    for m in (exp.metric_entries or [])
                ],
            }

            return web.json_response(result)

        except Exception as e:
            logging.exception("Failed to fetch experiment by ID")
            return web.json_response({"error": str(e)}, status=500)

        finally:
            session.close()


async def get_user_experiments(request: web.Request) -> web.Response:
    email = request.match_info.get("email", "").strip()
    email = email.upper()
    limit = int(request.query.get("limit", 20))
    offset = int(request.query.get("offset", 0))

    if not email:
        return web.json_response({"error": "Missing email"}, status=400)

    with get_db() as session:
        try:
            experiments = (
                session.query(ExperimentEvent)
                .join(ExperimentEvent.author)
                .filter(ExperimentAuthor.author_email == email)
                .order_by(ExperimentEvent.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            total = (
                session.query(ExperimentEvent)
                .join(ExperimentEvent.author)
                .filter(ExperimentAuthor.author_email == email)
                .count()
            )

            result = []
            for exp in experiments:
                cfg = exp.executed_config
                hw = exp.hardware
                result.append(
                    {
                        "id": exp.id,
                        "created_at": exp.created_at.isoformat()
                        if exp.created_at
                        else None,
                        "is_successful": exp.is_successful,
                        "total_exec_time": exp.total_exec_time,
                        "error_message": exp.error_message,
                        "config": {
                            "strategy": cfg.aggregation_strategy_keyword
                            if cfg
                            else None,
                            "dataset": cfg.dataset_keyword if cfg else None,
                            "num_rounds": cfg.num_rounds if cfg else None,
                            "num_clients": cfg.num_clients if cfg else None,
                            "num_malicious_clients": cfg.num_malicious_clients
                            if cfg
                            else None,
                        },
                        "hardware": {
                            "cpu": hw.cpu_name if hw else None,
                            "arch": hw.cpu_arch if hw else None,
                            "ram": hw.ram_amount if hw else None,
                            "gpu_name": hw.gpu_name if hw else None,
                            "gpu_ram": hw.gpu_ram if hw else None,
                            "is_executed_on": hw.is_executed_on if hw else None,
                        },
                    }
                )

            return web.json_response(
                {
                    "email": email,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "experiments": result,
                }
            )

        except Exception as e:
            logging.exception("Failed to fetch experiments by author")
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
            db_size_gb = (
                round(db_bytes / (1024**3), 3) if db_bytes is not None else "unknown"
            )

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

                last_created_at = dt.strftime("%B %d, %Y %I:%M%p %Z")
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


async def get_stats(request: web.Request) -> web.Response:
    """
    Return global experiment statistics as HTML.
    GET /experiments/stats
    """
    with get_db() as session:
        try:
            total = session.query(ExperimentEvent).count()
            successful = (
                session.query(ExperimentEvent)
                .filter(ExperimentEvent.is_successful == True)
                .count()
            )
            failed = (
                session.query(ExperimentEvent)
                .filter(ExperimentEvent.is_successful == False)
                .count()
            )

            strategy_rows = (
                session.query(
                    ExecutedConfig.aggregation_strategy_keyword,
                    func.count(ExperimentEvent.id).label("run_count"),
                    func.sum(
                        case((ExperimentEvent.is_successful == True, 1), else_=0)
                    ).label("successful_count"),
                )
                .join(ExperimentEvent.executed_config)
                .group_by(ExecutedConfig.aggregation_strategy_keyword)
                .order_by(func.count(ExperimentEvent.id).desc())
                .all()
            )

            dataset_rows = (
                session.query(
                    ExecutedConfig.dataset_keyword,
                    func.count(ExperimentEvent.id).label("run_count"),
                    func.sum(
                        case((ExperimentEvent.is_successful == True, 1), else_=0)
                    ).label("successful_count"),
                )
                .join(ExperimentEvent.executed_config)
                .group_by(ExecutedConfig.dataset_keyword)
                .order_by(func.count(ExperimentEvent.id).desc())
                .all()
            )

            author_rows = (
                session.query(
                    ExperimentAuthor.author_email,
                    ExperimentAuthor.full_author_name,
                    func.count(ExperimentEvent.id).label("run_count"),
                    func.sum(
                        case((ExperimentEvent.is_successful == True, 1), else_=0)
                    ).label("successful_count"),
                )
                .join(ExperimentEvent.author)
                .group_by(
                    ExperimentAuthor.author_email, ExperimentAuthor.full_author_name
                )
                .order_by(func.count(ExperimentEvent.id).desc())
                .all()
            )

            success_rate = round(successful / total * 100, 1) if total > 0 else 0

            def build_table(headers, rows):
                ths = "".join(f"<th>{h}</th>" for h in headers)
                trs = "".join(
                    "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                    for row in rows
                )
                return f"""
                <div class="table-wrap">
                    <table>
                        <thead><tr>{ths}</tr></thead>
                        <tbody>{trs}</tbody>
                    </table>
                </div>
                """

            strategy_table = build_table(
                ["Strategy", "Runs", "Successful", "Success Rate"],
                [
                    (
                        row.aggregation_strategy_keyword or "unknown",
                        row.run_count,
                        row.successful_count,
                        f"{round(row.successful_count / row.run_count * 100, 1) if row.run_count > 0 else 0}%",
                    )
                    for row in strategy_rows
                ],
            )

            dataset_table = build_table(
                ["Dataset", "Runs", "Successful", "Success Rate"],
                [
                    (
                        row.dataset_keyword or "unknown",
                        row.run_count,
                        row.successful_count,
                        f"{round(row.successful_count / row.run_count * 100, 1) if row.run_count > 0 else 0}%",
                    )
                    for row in dataset_rows
                ],
            )

            author_table = build_table(
                ["Author", "Email", "Runs", "Successful", "Success Rate"],
                [
                    (
                        row.full_author_name or "unknown",
                        row.author_email or "unknown",
                        row.run_count,
                        row.successful_count,
                        f"{round(row.successful_count / row.run_count * 100, 1) if row.run_count > 0 else 0}%",
                    )
                    for row in author_rows
                ],
            )

            html = f"""
            <html>
            <head>
                <title>InteFL Stats</title>
                <style>
                    body {{
                        margin: 0;
                        min-height: 100vh;
                        font-family: Arial, sans-serif;
                        background-color: #ffffff;
                        color: #333;
                        padding: 48px 32px;
                    }}
                    h1 {{
                        font-size: 48px;
                        margin-bottom: 6px;
                        color: #111;
                    }}
                    .subtitle {{
                        font-size: 18px;
                        color: #888;
                        margin-bottom: 40px;
                    }}
                    .summary {{
                        display: flex;
                        gap: 24px;
                        flex-wrap: wrap;
                        margin-bottom: 48px;
                    }}
                    .card {{
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 24px 32px;
                        min-width: 160px;
                        text-align: center;
                    }}
                    .card .value {{
                        font-size: 36px;
                        font-weight: bold;
                        color: #111;
                    }}
                    .card .label {{
                        font-size: 14px;
                        color: #888;
                        margin-top: 4px;
                    }}
                    .section {{
                        margin-bottom: 48px;
                    }}
                    .section-title {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #111;
                        margin-bottom: 16px;
                        padding-bottom: 8px;
                        border-bottom: 1px solid #eee;
                    }}
                    .table-wrap {{
                        overflow-x: auto;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 15px;
                    }}
                    thead tr {{
                        background: #f7f7f7;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        padding: 12px 20px;
                        text-align: left;
                        color: #555;
                        font-size: 13px;
                        text-transform: uppercase;
                        letter-spacing: 0.04em;
                        white-space: nowrap;
                    }}
                    tbody tr {{
                        border-bottom: 1px solid #f0f0f0;
                    }}
                    tbody tr:hover {{
                        background: #fafafa;
                    }}
                    td {{
                        padding: 10px 20px;
                        color: #444;
                    }}
                </style>
            </head>
            <body>
                <h1>InteFL Statistics</h1>
                <div class="subtitle">Overview of all recorded experiments</div>

                <div class="summary">
                    <div class="card">
                        <div class="value">{total}</div>
                        <div class="label">Total Experiments</div>
                    </div>
                    <div class="card">
                        <div class="value">{successful}</div>
                        <div class="label">Successful</div>
                    </div>
                    <div class="card">
                        <div class="value">{failed}</div>
                        <div class="label">Failed</div>
                    </div>
                    <div class="card">
                        <div class="value">{success_rate}%</div>
                        <div class="label">Success Rate</div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">By Strategy</div>
                    {strategy_table}
                </div>

                <div class="section">
                    <div class="section-title">By Dataset</div>
                    {dataset_table}
                </div>

                <div class="section">
                    <div class="section-title">By Author</div>
                    {author_table}
                </div>
            </body>
            </html>
            """

            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            logging.exception("Failed to fetch stats")
            return web.Response(
                text=f"<h2>Internal error: {e}</h2>",
                content_type="text/html",
                status=500,
            )

        finally:
            session.close()


async def compare_experiments(request: web.Request) -> web.Response:
    """
    Compare a specific metric across multiple experiments, aligned by round.
    GET /experiments/compare?ids=1,2,3&metric=accuracy_history&clients=3
    For per_round metrics: one row per round.
    For per_client metrics: one row per (round, client_number), filtered to clients 0..clients-1.
    """
    ids_param = request.query.get("ids", "")
    metric_name = request.query.get("metric", "").strip()
    clients_param = int(request.query.get("clients", 0))

    if not ids_param:
        return web.Response(
            text="<h2>Missing required query param: ids</h2>",
            content_type="text/html",
            status=400,
        )
    if not metric_name:
        return web.Response(
            text="<h2>Missing required query param: metric</h2>",
            content_type="text/html",
            status=400,
        )

    try:
        experiment_ids = [int(i.strip()) for i in ids_param.split(",") if i.strip()]
    except ValueError:
        return web.Response(
            text="<h2>ids must be a comma-separated list of integers</h2>",
            content_type="text/html",
            status=400,
        )

    with get_db() as session:
        try:
            # Resolve metric info
            metric_info = (
                session.query(MetricInfo).filter(MetricInfo.name == metric_name).first()
            )
            if metric_info is None:
                return web.Response(
                    text=f"<h2>Unknown metric: {metric_name}</h2>",
                    content_type="text/html",
                    status=404,
                )

            is_per_client = metric_info.type == "per_client"

            # Fetch entries
            query = session.query(MetricEntry).filter(
                MetricEntry.experiment_event_id.in_(experiment_ids),
                MetricEntry.metric_info_id == metric_info.id,
            )

            if is_per_client and clients_param > 0:
                # Filter to client_number 0..(clients_param-1) via join
                query = query.join(MetricEntry.client).filter(
                    ExperimentClient.client_number < clients_param
                )

            entries = query.order_by(MetricEntry.round_number).all()

            # Build lookup
            if is_per_client:
                # lookup[exp_id][(round, client_number)] = value
                lookup: dict = {eid: {} for eid in experiment_ids}
                row_keys = set()
                for e in entries:
                    client_num = e.client.client_number if e.client else None
                    key = (e.round_number, client_num)
                    lookup[e.experiment_event_id][key] = e.value
                    row_keys.add(key)
                row_keys = sorted(row_keys)
            else:
                # lookup[exp_id][round] = value
                lookup = {eid: {} for eid in experiment_ids}
                all_rounds = set()
                for e in entries:
                    lookup[e.experiment_event_id][e.round_number] = e.value
                    all_rounds.add(e.round_number)
                row_keys = sorted(all_rounds)

            # Fetch config labels
            configs = {}
            for eid in experiment_ids:
                exp = (
                    session.query(ExperimentEvent)
                    .filter(ExperimentEvent.id == eid)
                    .first()
                )
                cfg = exp.executed_config if exp else None
                configs[eid] = {
                    "strategy": cfg.aggregation_strategy_keyword if cfg else "?",
                    "dataset": cfg.dataset_keyword if cfg else "?",
                }

            # Build HTML table
            col_header = "Round" if not is_per_client else "Round / Client"

            header_cells = f"<th>{col_header}</th>" + "".join(
                f"<th>Exp {eid}<br><span class='sub'>{configs[eid]['strategy']} / {configs[eid]['dataset']}</span></th>"
                for eid in experiment_ids
            )

            rows_html = ""
            for key in row_keys:
                if is_per_client:
                    round_num, client_num = key
                    label = f"{round_num} / {client_num}"
                else:
                    label = str(key)

                cells = f"<td class='label'>{label}</td>"
                for eid in experiment_ids:
                    val = lookup[eid].get(key, None)
                    cells += f"<td>{round(val, 6) if val is not None else '<span class=null>—</span>'}</td>"
                rows_html += f"<tr>{cells}</tr>"

            html = f"""
            <html>
            <head>
                <title>Compare: {metric_name}</title>
                <style>
                    body {{
                        margin: 0;
                        min-height: 100vh;
                        font-family: Arial, sans-serif;
                        background-color: #ffffff;
                        color: #333;
                        padding: 48px 32px;
                    }}
                    .label-tag {{
                        font-size: 13px;
                        letter-spacing: 0.08em;
                        text-transform: uppercase;
                        color: #888;
                        margin-bottom: 8px;
                    }}
                    h1 {{
                        font-size: 36px;
                        font-weight: bold;
                        color: #111;
                        margin-bottom: 6px;
                    }}
                    .meta {{
                        margin-top: 6px;
                        font-size: 16px;
                        color: #666;
                        margin-bottom: 40px;
                    }}
                    .table-wrap {{
                        overflow-x: auto;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 15px;
                    }}
                    thead tr {{
                        background: #f7f7f7;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        padding: 14px 20px;
                        text-align: left;
                        color: #555;
                        font-size: 13px;
                        letter-spacing: 0.04em;
                        text-transform: uppercase;
                        white-space: nowrap;
                    }}
                    th .sub {{
                        display: block;
                        font-size: 11px;
                        color: #aaa;
                        margin-top: 3px;
                        text-transform: none;
                        letter-spacing: 0;
                    }}
                    tbody tr {{
                        border-bottom: 1px solid #f0f0f0;
                    }}
                    tbody tr:hover {{
                        background: #fafafa;
                    }}
                    td {{
                        padding: 10px 20px;
                        color: #444;
                        font-variant-numeric: tabular-nums;
                    }}
                    td.label {{
                        color: #111;
                        font-weight: 600;
                        white-space: nowrap;
                    }}
                    .null {{
                        color: #ccc;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="label-tag">Experiment Comparison</div>
                    <h1>{metric_name}</h1>
                    <div class="meta">
                        Experiments: {", ".join(str(i) for i in experiment_ids)}
                        &nbsp;·&nbsp; Type: {metric_info.type}
                        &nbsp;·&nbsp; Rows: {len(row_keys)}
                    </div>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead><tr>{header_cells}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            </body>
            </html>
            """

            return web.Response(text=html, content_type="text/html")

        except Exception as e:
            logging.exception("Failed to compare experiments")
            return web.Response(
                text=f"<h2>Internal error: {e}</h2>",
                content_type="text/html",
                status=500,
            )

        finally:
            session.close()


app = web.Application(middlewares=[auth_middleware])
app.router.add_get("/experiments/compare", compare_experiments)
app.router.add_get("/experiments/stats", get_stats)
app.router.add_get("/experiments/user/{email}", get_user_experiments)
app.router.add_get("/experiments/{id}", get_experiment_by_id)
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
