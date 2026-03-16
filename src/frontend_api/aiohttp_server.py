from __future__ import annotations

from aiohttp import web
from typing import Any, Dict

from src.frontend_api.execution_manager import generate_run_id, write_config_json, start_simulation_process
from src.frontend_api.log_manager import tail_lines, log_path
from src.config_loaders.config_loader import merge_and_validate_usecase_configs

# keep process handles in memory (optional; for status)
SIMULATION_PROCESSES: Dict[str, Any] = {}


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

    if not isinstance(received_config, dict):
        return web.json_response({"error": "config must be an object"}, status=400)

    errors = _validate_config_in_framework_backend(received_config)
    if errors:
        return web.json_response({"error": "validation_failed", "errors": errors}, status=400)

    run_id = generate_run_id()
    config_path = write_config_json(run_id, received_config)

    simulation_process = start_simulation_process(run_id, config_path, log_level=payload.get("log_level", "INFO"))
    SIMULATION_PROCESSES[run_id] = simulation_process

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


def create_app() -> web.Application:
    app = web.Application()

    app.router.add_get("/api/health", health)
    app.router.add_post("/api/validate_json_config", validate_json_config)
    app.router.add_post("/api/start_simulation_runner", start_simulation_runner)
    app.router.add_get("/api/logs", logs)
    app.router.add_get("/api/status", status)

    return app


def main() -> None:
    web.run_app(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
