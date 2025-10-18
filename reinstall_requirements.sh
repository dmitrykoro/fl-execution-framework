#!/bin/sh
# Deletes and recreates the virtual environment from requirements.txt

. "$(dirname "$0")/tests/scripts/common.sh"

find_python_interpreter

VENV_NAME=$(get_venv_name)

log_info "Removing existing '$VENV_NAME' directory..."
rm -rf "$VENV_NAME"

log_info "Creating new 'venv' virtual environment..."
run_python -m venv venv

setup_virtual_environment

# Set PYTHON_CMD to venv Python after activation
if [ -f "venv/Scripts/python.exe" ]; then
    PYTHON_CMD="venv/Scripts/python.exe"
    export PYTHON_CMD
    PYTHON_ARGS=""
    export PYTHON_ARGS
    log_info "Using venv Python: $PYTHON_CMD"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    export PYTHON_CMD
    PYTHON_ARGS=""
    export PYTHON_ARGS
    log_info "Using venv Python: $PYTHON_CMD"
fi

log_info "Upgrading pip..."
run_python -m pip install --upgrade pip

install_requirements
