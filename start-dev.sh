#!/bin/sh
# Dev server startup script for FL Execution Framework

set -eu

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/tests/scripts/common.sh"

# Navigate to project root
navigate_to_root

log_info "🚀 Starting FL Framework Development Servers..."

# Setup Python environment
find_python_interpreter
setup_virtual_environment

# Install Python API dependencies
if [ -f "requirements.txt" ]; then
    log_info "📦 Installing Python dependencies..."
    if pip install -q -r requirements.txt; then
        log_info "Python dependencies installed"
    else
        log_error "Failed to install Python dependencies"
        exit 1
    fi
else
    log_error "requirements.txt not found"
    exit 1
fi

# Install frontend dependencies
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    log_info "📦 Installing frontend dependencies..."
    (cd frontend && npm install)
    log_info "Frontend dependencies installed"
else
    log_error "Frontend directory or package.json not found"
    exit 1
fi

log_info "✅ Setup complete!"
echo ""
log_info "Starting servers..."
echo "  - API: http://127.0.0.1:8000"
echo "  - Frontend: http://localhost:5173"
echo ""
log_info "Press Ctrl+C to stop both servers"
echo ""

# Create log directory and files
mkdir -p tests/logs
API_LOG="tests/logs/api_dev_$(date +%Y%m%d_%H%M%S).log"
FRONTEND_LOG="tests/logs/frontend_dev_$(date +%Y%m%d_%H%M%S).log"
: > "$API_LOG"
: > "$FRONTEND_LOG"

# Start API in background with logging
uvicorn src.api.main:app --reload --port 8000 > "$API_LOG" 2>&1 &
API_PID=$!

# Start frontend in background with logging
(cd frontend && npm run dev > "../$FRONTEND_LOG" 2>&1) &
FRONTEND_PID=$!

# Wait for servers to start, then open browser
sleep 3
if command_exists xdg-open; then
    xdg-open http://localhost:5173 2>/dev/null || true
elif command_exists open; then
    open http://localhost:5173 2>/dev/null || true
elif command_exists start; then
    start http://localhost:5173 2>/dev/null || true
fi

# Trap Ctrl+C to kill both processes
cleanup() {
    echo ""
    log_info "🛑 Stopping servers..."
    kill $API_PID $FRONTEND_PID 2>/dev/null || true
    wait $API_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    log_info "Servers stopped. Logs saved to tests/logs/"
    exit 0
}
trap cleanup INT TERM

echo ""
log_info "📋 Tailing logs (Ctrl+C to stop)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Tail both logs with prefixes
tail -f "$API_LOG" "$FRONTEND_LOG" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        *"==> $API_LOG <=="*)
            echo ""
            echo "🔵 [API]"
            ;;
        *"==> $FRONTEND_LOG <=="*)
            echo ""
            echo "🟢 [FRONTEND]"
            ;;
        "")
            ;;
        *)
            echo "$line"
            ;;
    esac
done
