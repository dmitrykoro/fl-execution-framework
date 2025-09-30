#!/bin/sh
# Runs the simulation, setting up the environment if needed

. "$(dirname "$0")/tests/scripts/common.sh"

if [ -z "${VIRTUAL_ENV:-}" ] && ! [ -d "venv" ] && ! [ -d ".venv" ]; then
    log_warning "Virtual environment not found. Running reinstall_requirements.sh to create 'venv'..."
    ./reinstall_requirements.sh
fi

setup_virtual_environment
find_python_interpreter
setup_joblib_env

if [ ! -d "datasets/bloodmnist" ]; then
  log_info "Datasets not found. Starting download..."
  DATASET_URL="https://fl-dataset-storage.s3.us-east-1.amazonaws.com/datasets.tar"

  mkdir -p datasets
  _orig_dir="$(pwd)"
  cd datasets || exit 1

  if command_exists wget; then
    log_info "Downloading with wget..."
    wget "$DATASET_URL"
  else
    log_info "Downloading with Python..."
    "$PYTHON_CMD" -c "import urllib.request; print('Downloading datasets.tar...'); urllib.request.urlretrieve('$DATASET_URL', 'datasets.tar')"
  fi

  log_info "Extracting datasets..."
  tar -xf datasets.tar
  rm datasets.tar
  cd "$_orig_dir" || exit 1
fi

log_info "🚀 Initializing simulation..."
if "$PYTHON_CMD" -m src.simulation_runner; then
    echo ""
    show_simulation_output_info "out/"
else
    log_error "❌ Simulation failed. Check the logs above for details."
    exit 1
fi
