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

if command_exists wget; then
  echo "wget is installed"
else
  echo "wget command not found, exiting"
  exit 1
fi


# If venv does not exist, create one and install requirements
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating a new one."
    sh reinstall_requirements.sh
    source $VENV_DIR/bin/activate
    echo "Activated the newly created venv"
else
    echo "Found existing venv, switching to it..."
    source $VENV_DIR/bin/activate
    echo "Activated the existing venv"
fi


if [ ! -d "datasets/bloodmnist" ]; then
  echo "Datasets not found. Starting download..."
  pushd datasets
  wget https://fl-dataset-storage.s3.us-east-1.amazonaws.com/datasets.tar
  tar -xf datasets.tar
  rm datasets.tar
  popd
fi

echo "Initializing simulation..."
$PYTHON src/simulation_runner.py
