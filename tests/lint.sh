#!/bin/bash
# Lint, format, and test runner for your tests/ directory
# Run with: ./lint.sh
set -euo pipefail

# Step 1: go to repo root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# echo "📦 Installing requirements..."
# pip install -r requirements.txt

# Step 2: lint/format tests/ only (configs picked up from tests/)
echo "🔧 Running isort on tests/..."
isort tests

echo "⚫ Running black on tests/..."
black tests

echo "🔍 Running flake8 on tests/..."
flake8 tests --config=tests/.flake8

# # Step 3: run pytest with logging
# echo "🧪 Running pytest..."
# pytest -v --tb=short -s tests | tee pytest.log

echo "✅ All linting, formatting, and tests completed!"
