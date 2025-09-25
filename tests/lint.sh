#!/bin/bash
# Python code quality and testing script
# 
# Usage: ./lint.sh [--test] [--sonar]
# Prerequisites: run ./reinstall_requirements.sh in root first
#
# Default behavior:
#   - code linting and formatting (ruff)
#   - static type checking (mypy, pyright)
#
# Options:
#   --test  includes pytest, requirement reinstall, and simulation tests
#   --sonar includes pytest and SonarQube static analysis

# fail fast
set -euo pipefail

# cd to project root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# logging setup
LOG_DIR="tests/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/lint_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"
exec > >(tee "$LOG_FILE") 2>&1

# virtual environment detection and activation
VENV_ACTIVATED=false

# check for .venv first, then venv
if [ -d ".venv" ]; then
    echo "🔌 Found .venv directory, activating virtual environment..."
    if [ -f ".venv/Scripts/activate" ]; then
        source ".venv/Scripts/activate"
        VENV_ACTIVATED=true
    elif [ -f ".venv/bin/activate" ]; then
        source ".venv/bin/activate"
        VENV_ACTIVATED=true
    fi
elif [ -d "venv" ]; then
    echo "🔌 Found venv directory, activating virtual environment..."
    if [ -f "venv/Scripts/activate" ]; then
        source "venv/Scripts/activate"
        VENV_ACTIVATED=true
    elif [ -f "venv/bin/activate" ]; then
        source "venv/bin/activate"
        VENV_ACTIVATED=true
    fi
fi

if [ "$VENV_ACTIVATED" = false ]; then
    echo "⚠️  No virtual environment found (.venv or venv)."
    echo "   From the root directory, run ./reinstall_requirements.sh to create one."
    echo "   Continue without venv? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# parse args
TEST_MODE=false
SONAR_MODE=false

for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE=true
            ;;
        --sonar)
            SONAR_MODE=true
            ;;
    esac
done

# install test deps in test mode or sonar mode
if [[ "$TEST_MODE" == true || "$SONAR_MODE" == true ]]; then
    echo "📦 Installing test requirements..."
    pip install -e tests
fi

# check tool availability
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "⚠️  $1 not found. Install with: $2"
        return 1
    fi
    return 0
}

# install tools if needed
if ! check_tool "ruff" "pip install ruff"; then
    echo "📦 Installing ruff..."
    pip install ruff
fi

if ! check_tool "mypy" "pip install mypy"; then
    echo "📦 Installing mypy..."
    pip install mypy
fi

# ruff linting and formatting
echo "⚡ Running ruff check..."
ruff check --fix tests/

echo "⚡ Running ruff format..."
ruff format tests/

# type checking with mypy
echo "🔍 Running mypy..."
mypy tests/conftest.py tests/fixtures/ tests/integration/ tests/unit/ tests/performance/ --config-file=tests/pyproject.toml

# additional type checking with pyright
if check_tool "pyright" "npm install -g pyright"; then
    echo "🔍 Running pyright..."
    pyright tests/
else
    echo "⚠️  Pyright not available. Install with: npm install -g pyright"
    echo "📦 Attempting to install pyright via npm..."
    if command -v npm &> /dev/null; then
        npm install -g pyright
        if check_tool "pyright" "npm install -g pyright"; then
            echo "🔍 Running pyright..."
            pyright tests/
        else
            echo "⚠️  Pyright installation failed. Skipping pyright check."
        fi
    else
        echo "⚠️  npm not available. Skipping pyright installation."
    fi
fi

# pytest with coverage
if [[ "$TEST_MODE" == true ]]; then
    echo "🧪 Running pytest..."
    # clear any existing coverage data
    coverage erase

    # run tests with coverage accumulation
    PYTHONPATH=. coverage run --source=src -m pytest tests/unit/ | tee tests/logs/pytest_unit.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/integration/ -s | tee tests/logs/pytest_integration.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/performance/ | tee tests/logs/pytest_performance.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/test_setup.py | tee tests/logs/pytest_setup.log

    # generate combined coverage reports
    coverage xml
    coverage html
    coverage report --skip-covered

    # check all pytest logs for failures
    if grep -q "FAILED" tests/logs/pytest_*.log; then
        echo "❌ Some tests failed. Check tests/logs/pytest_*.log for details."
        exit 1
    fi

    # test reinstall script
    echo "🔄 Testing reinstall_requirements.sh..."
    ./reinstall_requirements.sh

    # test simulation script
    echo "🚀 Testing run_simulation.sh..."
    ./run_simulation.sh

    echo
    echo "✅ All linting, formatting, and tests completed!"
fi

# pytest with coverage for SonarQube analysis
if [[ "$SONAR_MODE" == true ]]; then
    echo "🧪 Running pytest with coverage for SonarQube analysis..."
    # clear any existing coverage data
    coverage erase

    # run tests with coverage accumulation
    PYTHONPATH=. coverage run --source=src -m pytest tests/unit/ | tee tests/logs/pytest_unit.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/integration/ -s | tee tests/logs/pytest_integration.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/performance/ | tee tests/logs/pytest_performance.log
    PYTHONPATH=. coverage run --source=src --append -m pytest tests/test_setup.py | tee tests/logs/pytest_setup.log

    # generate combined coverage reports
    coverage xml
    coverage html
    coverage report --skip-covered

    # check all pytest logs for failures
    if grep -q "FAILED" tests/logs/pytest_*.log; then
        echo "❌ Some tests failed. Check tests/logs/pytest_*.log for details."
        exit 1
    fi

    # run SonarQube analysis
    ./tests/sonar.sh
fi

# final status message
if [[ "$TEST_MODE" == false && "$SONAR_MODE" == true ]]; then
    echo "✅ Linting, type checking, and SonarQube analysis completed!"
elif [[ "$TEST_MODE" == false && "$SONAR_MODE" == false ]]; then
    echo "✅ Linting, formatting, and type checking completed!"
fi

# print summary
echo ""
echo "🏁 Summary:"
echo "   ⚡ Used ruff for linting and formatting"
echo "   🔍 Ran mypy for type checking"
if check_tool "pyright" "npm install -g pyright" > /dev/null 2>&1; then
    echo "   🔍 Ran pyright for additional type checking"
fi
if [[ "$SONAR_MODE" == true ]]; then
    echo "   🔍 Ran SonarQube analysis"
fi
echo "   📝 Log saved to: $LOG_FILE"
