#!/bin/bash
# Deletes and recreates the virtual environment from requirements.txt

# Source common utilities
source "$(dirname "$0")/tests/scripts/common.sh"

# Find a compatible Python interpreter
find_python_interpreter

# Determine the name of the virtual environment directory
VENV_NAME="venv"
if [ -d ".venv" ]; then
    VENV_NAME=".venv"
fi

log_info "Removing existing '$VENV_NAME' directory..."
rm -rf "$VENV_NAME"

log_info "Creating new 'venv' virtual environment..."
$PYTHON_CMD -m venv venv

# Activate the new environment
setup_virtual_environment

log_info "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

log_info "Installing requirements from requirements.txt..."
pip install -r requirements.txt
log_info "Requirements installed successfully."
