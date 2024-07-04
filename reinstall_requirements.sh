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

if command_exists pip3; then
    PIP=pip3
elif command_exists pip; then
    PIP=pip
else
    echo "pip is not installed. Exiting."
    exit 1
fi

echo "Removing existing venv..."
rm -rf $VENV_DIR

echo "Creating new venv..."
$PYTHON -m venv $VENV_DIR

source $VENV_DIR/bin/activate
echo "Installing requirements..."

$PIP install -r requirements.txt
echo "Requirements installed."
