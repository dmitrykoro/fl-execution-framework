#!/bin/bash
# Lint script with ruff and pyright
# Run: ./lint.sh [--full] [--sonar]
# Prerequisites: ./reinstall_requirements.sh
#
# Default: ruff (linting + formatting), mypy, pyright
# --full: adds pytest, reinstall test, simulation test
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
FULL_MODE=false
SONAR_MODE=false

for arg in "$@"; do
    case $arg in
        --full)
            FULL_MODE=true
            ;;
        --sonar)
            SONAR_MODE=true
            ;;
    esac
done

# install test deps in full mode
if [[ "$FULL_MODE" == true ]]; then
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
    if check_tool "sonar-scanner" "npm install -g sonar-scanner"; then
        echo "🔍 Running SonarQube analysis..."

        # Create basic sonar-project.properties if it doesn't exist
        if [[ ! -f "sonar-project.properties" ]]; then
            cat > sonar-project.properties << EOF
sonar.projectKey=fl-execution-framework
sonar.projectName=FL Execution Framework
sonar.projectVersion=1.0
sonar.sources=tests
sonar.python.coverage.reportPaths=coverage.xml
sonar.exclusions=**/logs/**,**/temp/**,**/__pycache__/**
EOF
            echo "📄 Created sonar-project.properties"
        fi

        sonar-scanner
    else
        echo "⚠️  SonarQube Scanner not available. Install with: npm install -g sonar-scanner"
    fi
fi

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
