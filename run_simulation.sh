#!/bin/bash

VENV_DIR="venv"

command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

if command_exists python3.10; then
    PYTHON=python3.10
else
    echo "Python3.10 is not installed. Exiting."
    exit 1
fi

# If venv does not exist, create one and install requirements
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating a new one."
    sh reinstall_requirements.sh
else
    echo "Found existing venv, switching to it..."
    source $VENV_DIR/bin/activate
    echo "Activated the existing venv"
fi

$PYTHON src/simulation_runner.py
