#!/bin/bash
# Runs the simulation, setting up the environment if needed

# Source common utilities
source "$(dirname "$0")/tests/scripts/common.sh"

# Set up the virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]] && ! [ -d "venv" ] && ! [ -d ".venv" ]; then
    log_warning "Virtual environment not found. Running reinstall_requirements.sh to create 'venv'..."
    ./reinstall_requirements.sh
fi

# Activate environment and find python
setup_virtual_environment
find_python_interpreter

# Check for datasets and download if missing
if [ ! -d "datasets/bloodmnist" ]; then
  log_info "Datasets not found. Starting download..."
  DATASET_URL="https://fl-dataset-storage.s3.us-east-1.amazonaws.com/datasets.tar"
  
  # Create datasets directory
  mkdir -p datasets
  pushd datasets > /dev/null

  if command_exists wget; then
    log_info "Downloading with wget..."
    wget "$DATASET_URL"
  else
    log_info "Downloading with Python..."
    $PYTHON_CMD -c "import urllib.request; print('Downloading datasets.tar...'); urllib.request.urlretrieve('$DATASET_URL', 'datasets.tar')"
  fi

  log_info "Extracting datasets..."
  tar -xf datasets.tar
  rm datasets.tar
  popd > /dev/null
fi

log_info "🚀 Initializing simulation..."
$PYTHON_CMD -m src.simulation_runner
