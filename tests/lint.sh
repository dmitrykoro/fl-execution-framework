#!/bin/bash
# Python code quality and testing script
# Usage: ./lint.sh [--test] [--sonar]

# Source common utilities and navigate to project root
source "$(dirname "$0")/scripts/common.sh"
navigate_to_root

# Logging setup
LOG_DIR="tests/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lint_$(date +%Y%m%d_%H%M%S).log"
log_info "📝 Logging to: $LOG_FILE"
exec > >(tee "$LOG_FILE") 2>&1

# Environment setup
setup_virtual_environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_error "No virtual environment activated. Please run ./reinstall_requirements.sh first."
    exit 1
fi

# Argument parsing
TEST_MODE=false
SONAR_MODE=false
for arg in "$@"; do
    case $arg in
        --test) TEST_MODE=true ;;
        --sonar) SONAR_MODE=true ;;
    esac
done

# Install dependencies if running tests or sonar
if [[ "$TEST_MODE" == true || "$SONAR_MODE" == true ]]; then
    log_info "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Code quality checks
log_info "⚡ Running ruff check..."
ruff check --fix tests/

log_info "⚡ Running ruff format..."
ruff format tests/

log_info "🔍 Running mypy..."
mypy tests/ --config-file=tests/pyproject.toml

if command_exists pyright; then
    log_info "🔍 Running pyright..."
    pyright tests/
else
    log_warning "Pyright not found. Skipping. To install: npm install -g pyright"
fi

# Test execution function
run_pytest_suite() {
    log_info "🧪 Running pytest suite with coverage..."
    coverage erase
    PYTHONPATH=. coverage run --source=src -m pytest tests/

    log_info "📊 Generating coverage reports..."
    coverage xml -o "$LOG_DIR/coverage.xml"
    coverage html -d "$LOG_DIR/coverage_html"
    coverage report --skip-covered
}

# Main logic
if [[ "$TEST_MODE" == true ]]; then
    run_pytest_suite
fi

if [[ "$SONAR_MODE" == true ]]; then
    [[ "$TEST_MODE" == false ]] && run_pytest_suite
    log_info "🔍 Running SonarQube analysis..."
    ./tests/scripts/sonar.sh
fi

# Final summary
echo ""
log_info "🏁 Linting and testing process finished."
log_info "📝 Full log saved to: $LOG_FILE"
if [[ "$TEST_MODE" == true || "$SONAR_MODE" == true ]]; then
    log_info "📊 Coverage reports saved to: $LOG_DIR/coverage.xml and $LOG_DIR/coverage_html/"
fi
