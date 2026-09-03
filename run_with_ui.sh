#!/bin/sh
# Runs the InteFL web UI: the aiohttp API server plus the React frontend.
# Both are stopped together when this script exits.

. "$(dirname "$0")/shell_scripts/common.sh"

# Allow the script to be started from anywhere.
cd "$(cd "$(dirname "$0")" && pwd)"

FRONTEND_DIR="frontend"
API_LOG="logs/ui_backend.log"
UI_PORT="${UI_PORT:-5173}"

# Fixed: the Vite dev proxy in frontend/vite.config.ts forwards /api to this port.
API_PORT=8000

API_PID=""
UI_PID=""

stop_child() {
    # $1: pid, $2: name
    if [ -n "$1" ] && kill -0 "$1" 2>/dev/null; then
        log_info "Stopping the $2..."
        kill "$1" 2>/dev/null || true
        wait "$1" 2>/dev/null || true
    fi
}

stop_everything() {
    stop_child "$UI_PID" "frontend"
    stop_child "$API_PID" "API server"
}

trap stop_everything EXIT INT TERM

# ---- Python environment ----
if ! ensure_virtual_environment; then
    log_warning "Running reinstall_requirements.sh to create 'venv'..."
    ./reinstall_requirements.sh
    setup_virtual_environment
fi

find_python_interpreter
setup_joblib_env

# ---- Frontend dependencies ----
if ! command_exists npm; then
    log_error "npm was not found. Install Node.js 18 or newer from https://nodejs.org and re-run this script."
    exit 1
fi

VITE_BIN="$FRONTEND_DIR/node_modules/.bin/vite"

if [ ! -x "$VITE_BIN" ]; then
    log_info "Installing frontend dependencies (first run only, this takes a minute)..."
    (cd "$FRONTEND_DIR" && npm install)
fi

if [ ! -x "$VITE_BIN" ]; then
    log_error "Frontend dependencies are missing: $VITE_BIN was not created by 'npm install'."
    exit 1
fi

# ---- Experiment database credentials ----
# Resolved here, in the foreground, so that a prompt for the experiment server token
# reaches the terminal. The API server is started in the background afterwards and
# picks up the answer, as do the simulations it spawns.
log_info "Checking experiment database configuration..."
run_python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
from src.experiment_database.remote_access import initialize_remote_access
initialize_remote_access()
"

# ---- API server ----
mkdir -p "$(dirname "$API_LOG")"
log_info "Starting the API server on http://127.0.0.1:$API_PORT ..."

if [ -n "${PYTHON_ARGS:-}" ]; then
    "$PYTHON_CMD" $PYTHON_ARGS -u -m src.frontend_api.aiohttp_server >>"$API_LOG" 2>&1 &
else
    "$PYTHON_CMD" -u -m src.frontend_api.aiohttp_server >>"$API_LOG" 2>&1 &
fi

API_PID=$!

waited=0
while [ "$waited" -lt 30 ]; do
    if run_python -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$API_PORT/api/health', timeout=1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        break
    fi

    if ! kill -0 "$API_PID" 2>/dev/null; then
        log_error "The API server exited during startup. Last lines of $API_LOG:"
        tail -n 20 "$API_LOG" >&2
        exit 1
    fi

    waited=$((waited + 1))
    sleep 1
done

if [ "$waited" -ge 30 ]; then
    log_error "The API server did not become ready within 30s. See $API_LOG."
    exit 1
fi

log_info "API server is ready (log: $API_LOG)."

# ---- Frontend ----
log_info ""
log_info "Open the UI at: http://localhost:$UI_PORT"
log_info "Press Ctrl+C to stop the UI and the API server."
log_info ""

(cd "$FRONTEND_DIR" && exec ./node_modules/.bin/vite --port "$UI_PORT" --strictPort) &
UI_PID=$!

wait "$UI_PID"
