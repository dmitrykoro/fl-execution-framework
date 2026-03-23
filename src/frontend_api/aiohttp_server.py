from __future__ import annotations

import aiohttp
import logging
import asyncio

from aiohttp import web
from typing import Any, Dict
from pathlib import Path

from src.frontend_api.execution_manager import generate_run_id, write_config_json, start_simulation_process
from src.frontend_api.log_manager import tail_lines, log_path
from src.frontend_api.agentic_configuration_manager import AgenticConfigurationManager
from src.frontend_api.db_analysis_manager import DatabaseAnalysisManager
from src.config_loaders.config_loader import merge_and_validate_usecase_configs
from src.utils.common import obtain_env_var

# keep process handles in memory (optional; for status)
SIMULATION_PROCESSES: Dict[str, Any] = {}
RUN_CONFIG_PATHS: Dict[str, str] = {}
AGENT_MANAGER = AgenticConfigurationManager()
DB_ANALYSIS_MANAGER = DatabaseAnalysisManager()


def _auth_headers() -> dict:
    """Common auth headers for experiment server requests."""
    return {
        "Authorization": f"Bearer {obtain_env_var('EXPERIMENT_SERVER_TOKEN')}",
        "Accept": "application/json",
    }


def _optional_session_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null", "undefined"}:
        return None
    return normalized


def _validate_config_in_framework_backend(received_config: Dict[str, Any]) -> list[str]:
    validation_errors = []

    try:
        merge_and_validate_usecase_configs(config_dict=received_config)

    except Exception as e:  # noqa
        validation_errors.append(str(e))
        return validation_errors

    return validation_errors


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def validate_json_config(request: web.Request) -> web.Response:
    payload = await request.json()
    received_config = payload.get("config")

    if not isinstance(received_config, dict):
        return web.json_response({"ok": False, "errors": ["config must be an object"]}, status=400)

    validation_errors = _validate_config_in_framework_backend(received_config)

    return web.json_response({"ok": len(validation_errors) == 0, "errors": validation_errors})


async def start_simulation_runner(request: web.Request) -> web.Response:
    payload = await request.json()
    received_config = payload.get("config")
    session_id = payload.get("agent_session_id")

    if not isinstance(received_config, dict):
        return web.json_response({"error": "config must be an object"}, status=400)

    errors = _validate_config_in_framework_backend(received_config)
    if errors:
        return web.json_response({"error": "validation_failed", "errors": errors}, status=400)

    run_id = generate_run_id()
    config_path = write_config_json(run_id, received_config)
    RUN_CONFIG_PATHS[run_id] = str(config_path)

    simulation_process = start_simulation_process(run_id, config_path, log_level=payload.get("log_level", "INFO"))
    SIMULATION_PROCESSES[run_id] = simulation_process

    if session_id:
        try:
            AGENT_MANAGER.attach_run(session_id, run_id)
        except KeyError:
            logging.warning("Tried to attach unknown AI agent session %s to run %s", session_id, run_id)

    return web.json_response({"run_id": run_id})


async def logs(request: web.Request) -> web.Response:
    run_id = request.query.get("run_id")
    if not run_id:
        return web.json_response({"error": "run_id query param required"}, status=400)

    lines = tail_lines(log_path(run_id), max_lines=400)
    return web.json_response({"lines": lines})


async def status(request: web.Request) -> web.Response:
    run_id = request.query.get("run_id")
    if not run_id:
        return web.json_response({"error": "run_id query param required"}, status=400)

    proc = SIMULATION_PROCESSES.get(run_id)
    if proc is None:
        # maybe server restarted; logs still exist
        return web.json_response({"run_id": run_id, "status": "UNKNOWN"})

    code = proc.poll()
    if code is None:
        return web.json_response({"run_id": run_id, "status": "RUNNING"})

    return web.json_response({"run_id": run_id, "status": "EXITED", "exit_code": code})


async def get_experiments(request: web.Request) -> web.Response:
    """
    Retrieve experiments from the experiment database server
    and return them to the frontend.
    """

    experiment_server_url = obtain_env_var("EXPERIMENT_SERVER_URL")
    page = request.query.get("page")
    page_size = request.query.get("page_size")
    params = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{experiment_server_url}/experiments",
                    params=params,
                    headers=_auth_headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_text = await resp.text()

                if resp.status != 200:
                    logging.error(
                        f"Failed to retrieve experiments: ({resp.status}): {response_text}"
                    )
                    return web.json_response(
                        {
                            "success": False,
                            "message": "Failed to retrieve experiments from remote DB server",
                            "status": resp.status,
                            "details": response_text,
                        },
                        status=resp.status,
                    )

                try:
                    response_json = await resp.json()
                except aiohttp.ContentTypeError:
                    logging.error(
                        f"Remote DB server returned non-JSON response: {response_text}"
                    )
                    return web.json_response(
                        {
                            "success": False,
                            "message": "Remote DB server returned invalid JSON",
                            "details": response_text,
                        },
                        status=502,
                    )

                logging.info("Successfully retrieved experiment data from remote DB")
                return web.json_response(response_json, status=200)

    except asyncio.TimeoutError:
        logging.error("Timeout while retrieving experiments from remote DB server")
        return web.json_response(
            {
                "success": False,
                "message": "Timeout while contacting remote DB server",
            },
            status=504,
        )

    except aiohttp.ClientError as e:
        logging.error(f"HTTP client error while retrieving experiments: {e}")
        return web.json_response(
            {
                "success": False,
                "message": "HTTP client error while contacting remote DB server",
                "details": str(e),
            },
            status=502,
        )

    except Exception as e:
        logging.exception(f"Unexpected error while retrieving experiments: {e}")
        return web.json_response(
            {
                "success": False,
                "message": "Unexpected error while retrieving experiments",
                "details": str(e),
            },
            status=500,
        )


async def agent_start(request: web.Request) -> web.Response:
    payload = await request.json()
    text = str(payload.get("text", "")).strip()
    session_id = _optional_session_id(payload.get("session_id"))
    if not text:
        return web.json_response({"error": "text is required"}, status=400)

    session = await AGENT_MANAGER.start_session(text, session_id=session_id)
    return web.json_response(AGENT_MANAGER._snapshot(session), status=202)


async def agent_refine(request: web.Request) -> web.Response:
    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()
    text = str(payload.get("text", "")).strip()
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)
    if not text:
        return web.json_response({"error": "text is required"}, status=400)

    try:
        session = await AGENT_MANAGER.refine_session(session_id, text)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    return web.json_response(AGENT_MANAGER._snapshot(session), status=202)


async def agent_poll(request: web.Request) -> web.Response:
    session_id = str(request.query.get("session_id", "")).strip()
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    try:
        since = int(request.query.get("since", "0"))
    except ValueError:
        return web.json_response({"error": "since must be an integer"}, status=400)

    try:
        snapshot = await AGENT_MANAGER.wait_for_update(session_id, since)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    return web.json_response(snapshot)


async def agent_sessions(request: web.Request) -> web.Response:
    items = await AGENT_MANAGER.list_sessions()
    return web.json_response({"items": items})


async def agent_get_session(request: web.Request) -> web.Response:
    session_id = _optional_session_id(request.query.get("session_id"))
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    try:
        session = await AGENT_MANAGER.load_session(session_id)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=502)

    return web.json_response(AGENT_MANAGER._snapshot(session))


async def agent_analyze_results(request: web.Request) -> web.Response:
    payload = await request.json()
    session_id = str(payload.get("session_id", "")).strip()
    run_id = str(payload.get("run_id", "")).strip()

    if not session_id or not run_id:
        return web.json_response({"error": "session_id and run_id are required"}, status=400)

    proc = SIMULATION_PROCESSES.get(run_id)
    simulation_crashed = False
    if proc is not None:
        code = proc.poll()
        if code is None:
            return web.json_response({"error": "run is still active"}, status=409)
        simulation_crashed = code != 0

    config_path = RUN_CONFIG_PATHS.get(run_id)
    if not config_path:
        return web.json_response({"error": "Unknown run_id config"}, status=404)

    results_dirname = str(Path("out") / run_id)
    try:
        session = await AGENT_MANAGER.analyze_results(
            session_id=session_id,
            run_id=run_id,
            results_dirname=results_dirname,
            generated_strategy_config_json_filepath=config_path,
            simulation_crashed=simulation_crashed,
        )
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    return web.json_response(AGENT_MANAGER._snapshot(session), status=202)


async def analyze_db_start(request: web.Request) -> web.Response:
    payload = await request.json()
    question = str(payload.get("question", "")).strip()
    session_id = _optional_session_id(payload.get("session_id"))
    if not question:
        return web.json_response({"error": "question is required"}, status=400)

    session = await DB_ANALYSIS_MANAGER.start_session(question, session_id=session_id)
    return web.json_response(DB_ANALYSIS_MANAGER._snapshot(session), status=202)


async def analyze_db_sessions(request: web.Request) -> web.Response:
    items = await DB_ANALYSIS_MANAGER.list_sessions()
    return web.json_response({"items": items})


async def analyze_db_get_session(request: web.Request) -> web.Response:
    session_id = _optional_session_id(request.query.get("session_id"))
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    try:
        session = await DB_ANALYSIS_MANAGER.load_session(session_id)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=502)

    return web.json_response(DB_ANALYSIS_MANAGER._snapshot(session))


async def analyze_db_poll(request: web.Request) -> web.Response:
    session_id = str(request.query.get("session_id", "")).strip()
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    try:
        since = int(request.query.get("since", "0"))
    except ValueError:
        return web.json_response({"error": "since must be an integer"}, status=400)

    try:
        snapshot = await DB_ANALYSIS_MANAGER.wait_for_update(session_id, since)
    except KeyError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    return web.json_response(snapshot)


def create_app() -> web.Application:
    app = web.Application()

    app.router.add_get("/api/health", health)
    app.router.add_post("/api/validate_json_config", validate_json_config)
    app.router.add_post("/api/start_simulation_runner", start_simulation_runner)
    app.router.add_get("/api/get_experiments", get_experiments)
    app.router.add_post("/api/agent/start", agent_start)
    app.router.add_post("/api/agent/refine", agent_refine)
    app.router.add_get("/api/agent/sessions", agent_sessions)
    app.router.add_get("/api/agent/session", agent_get_session)
    app.router.add_get("/api/agent/poll", agent_poll)
    app.router.add_post("/api/agent/analyze_results", agent_analyze_results)
    app.router.add_post("/api/analyze_db/start", analyze_db_start)
    app.router.add_get("/api/analyze_db/sessions", analyze_db_sessions)
    app.router.add_get("/api/analyze_db/session", analyze_db_get_session)
    app.router.add_get("/api/analyze_db/poll", analyze_db_poll)
    app.router.add_get("/api/logs", logs)
    app.router.add_get("/api/status", status)

    return app


def main() -> None:
    web.run_app(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
