#!/bin/bash
# Lint, format, and test the codebase
# Run with: ./lint.sh

# exit immediately if a command fails
set -euo pipefail

# navigate to the project root directory
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# install dependencies from tests/pyproject.toml
echo "📦 Installing test requirements..."
pip install -e tests

# sort imports
echo "🔧 Running isort..."
isort tests

# format code with black
echo "⚫ Running black..."
black tests

# lint code with flake8
echo "🔍 Running flake8..."
flake8 tests --config=tests/.flake8

# run mypy type checking on test infrastructure
echo "🔍 Running mypy..."
mypy tests/conftest.py tests/fixtures/ tests/integration/ tests/unit/ tests/performance/ --config-file=tests/pyproject.toml

# run pytest with logging to a pytest.log file
echo "🧪 Running pytest..."
python -m pytest -v --tb=short -s tests | tee pytest.log
if grep -q "FAILED" pytest.log; then
    echo "❌ Some tests failed. Check pytest.log for details."
    exit 1
fi

echo "✅ All linting, formatting, and tests completed!"
