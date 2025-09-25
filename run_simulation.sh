#!/bin/bash
# Runs the simulation, setting up the environment if needed.

# Exit immediately on error.
set -e

VENV_DIR=""

command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

activate_venv () {
    # Handle Windows and Unix-like activation paths.
    if [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
}

# Find a compatible Python 3.9+ interpreter, checking newer versions first.
PYTHON=""
for version in python python3.11 python3.10 python3.9 python3; do
    if command_exists $version; then
        VERSION_CHECK=$($version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON=$version
            break
        fi
    fi
done

# Fallback to 'py' launcher on Windows.
if [ -z "$PYTHON" ] && command_exists py; then
    for version in "-3.11" "-3.10" "-3.9" "-3"; do
        VERSION_CHECK=$(py $version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON="py $version"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "Python 3.9+ is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Use wget for downloads, fall back to Python for portability.
DOWNLOAD_METHOD="python"
if command_exists wget; then
  DOWNLOAD_METHOD="wget"
fi

# Find and activate virtual environment.
# Checks .venv first, then venv, creates .venv if neither exists.
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
fi

if [ -n "$VENV_DIR" ]; then
    echo "Found existing venv in '$VENV_DIR', activating..."
    activate_venv
else
    echo "Virtual environment not found, creating '.venv'..."
    sh reinstall_requirements.sh
    VENV_DIR=".venv"
    activate_venv
fi

# Check for a sample dataset to see if download is needed.
if [ ! -d "datasets/bloodmnist" ]; then
  echo "Datasets not found. Starting download..."
  pushd datasets > /dev/null

  if [ "$DOWNLOAD_METHOD" = "wget" ]; then
    wget https://fl-dataset-storage.s3.us-east-1.amazonaws.com/datasets.tar
  else
    $PYTHON -c "import urllib.request; urllib.request.urlretrieve('https://fl-dataset-storage.s3.us-east-1.amazonaws.com/datasets.tar', 'datasets.tar')"
  fi

  tar -xf datasets.tar
  rm datasets.tar
  popd > /dev/null
fi

echo "Initializing simulation..."
$PYTHON -m src.simulation_runner
