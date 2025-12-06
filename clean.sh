#!/bin/sh
# Cleans the out/ directory, preserving .gitkeep

find out -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
echo "Cleaned out/ directory."

# Clean logs/ directory entirely (preserving directory itself)
if [ -d "logs" ]; then
    find logs -mindepth 1 -exec rm -rf {} +
    echo "Cleaned logs/ directory."
fi

# Clean tests/logs/ directory entirely (preserving directory itself)
if [ -d "tests/logs" ]; then
    find tests/logs -mindepth 1 -exec rm -rf {} +
    echo "Cleaned tests/logs/ directory."
fi

# Clean tool caches (safe to remove, just re-generates on next run)
rm -rf .mypy_cache .pytest_cache .ruff_cache
echo "Cleaned tool caches (.mypy_cache, .pytest_cache, .ruff_cache)."
