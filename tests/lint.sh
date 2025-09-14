#!/bin/bash

# Test linting and formatting script
# Run with: ./lint.sh

set -e  # Exit on first error

echo "🔧 Running isort..."
isort .

echo "⚫ Running black..."
black .

echo "🔍 Running flake8..."
flake8 --ignore=E501,W503,E203 .

echo "✅ All linting and formatting completed!"