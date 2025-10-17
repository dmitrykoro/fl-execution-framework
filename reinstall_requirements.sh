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

log_info "Upgrading pip..."
run_python -m pip install --upgrade pip

install_requirements
install_requirements "src/api/requirements.txt"
