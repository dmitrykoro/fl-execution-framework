#!/bin/bash
# Run: ./lint.sh [--full]
# Prerequisites: ./reinstall_requirements.sh
#
# Default: isort, black, flake8, mypy
# --full: adds pytest, reinstall test, simulation test

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

# activate venv
if [ -f "venv/Scripts/activate" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found."
    echo "   Run ./reinstall_requirements.sh first to create one."
    echo "   Continue without venv? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# parse args
FULL_MODE=false
if [[ "${1:-}" == "--full" ]]; then
    FULL_MODE=true
fi

# install test deps in full mode
if [[ "$FULL_MODE" == true ]]; then
    echo "📦 Installing test requirements..."
    pip install -e tests
fi

# import sorting
echo "🔧 Running isort..."
isort tests

# code formatting
echo "⚫ Running black..."
black tests

# linting
echo "🔍 Running flake8..."
flake8 tests --config=tests/.flake8

# type checking
echo "🔍 Running mypy..."
mypy tests/conftest.py tests/fixtures/ tests/integration/ tests/unit/ tests/performance/ --config-file=tests/pyproject.toml

# pytest in full mode
if [[ "$FULL_MODE" == true ]]; then
    echo "🧪 Running pytest..."
    pytest -v --tb=short -s tests | tee pytest.log
    if grep -q "FAILED" pytest.log; then
        echo "❌ Some tests failed. Check pytest.log for details."
        exit 1
    fi

    # test reinstall script
    echo "🔄 Testing reinstall_requirements.sh..."
    ./reinstall_requirements.sh

    # test simulation script
    echo "🚀 Testing run_simulation.sh..."
    ./run_simulation.sh

    echo "✅ All linting, formatting, and tests completed!"
else
    echo "✅ Linting and formatting completed!"
fi
