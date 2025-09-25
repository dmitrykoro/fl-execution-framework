#!/bin/bash
# Lint script with ruff and pyright
# Run: ./lint.sh [--test] [--sonar]
# Prerequisites: ./reinstall_requirements.sh
#
# Default: ruff (linting + formatting), mypy, pyright
# --test: adds pytest, reinstall test, simulation test
# --sonar: adds SonarQube analysis

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

# Dynamic virtual environment detection and activation
VENV_ACTIVATED=false

# Check for .venv first, then venv
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

# install test deps in full mode
if [[ "$TEST_MODE" == true ]]; then
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

# Install tools if needed
if ! check_tool "ruff" "pip install ruff"; then
    echo "📦 Installing ruff..."
    pip install ruff
fi

if ! check_tool "mypy" "pip install mypy"; then
    echo "📦 Installing mypy..."
    pip install mypy
fi

# Ruff linting and formatting (replaces isort + flake8 + black)
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

# SonarQube analysis
if [[ "$SONAR_MODE" == true ]]; then
    ./tests/sonar.sh
fi

# pytest in test mode
if [[ "$TEST_MODE" == true ]]; then
    echo "🧪 Running pytest..."
    # Run unit tests in parallel, integration tests serially
    pytest -n auto tests/unit/ -v --tb=short | tee tests/logs/pytest_unit.log
    pytest -n 0 tests/integration/ -v --tb=short -s | tee tests/logs/pytest_integration.log
    pytest tests/performance/ -v --tb=short | tee tests/logs/pytest_performance.log
    pytest tests/test_setup.py -v --tb=short | tee tests/logs/pytest_setup.log
    # Check all pytest logs for failures
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
else
    if [[ "$SONAR_MODE" == true ]]; then
        echo "✅ Linting, type checking, and SonarQube analysis completed!"
    else
        echo "✅ Linting, formatting, and type checking completed!"
    fi
fi

# Print summary
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
