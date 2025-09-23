#!/bin/bash
# Deletes and recreates the virtual environment from requirements.txt.

# Exit immediately on error.
set -e

# Determine virtual environment directory
# Check for existing naming convention, default to .venv for new installations
VENV_DIR=""
if [ -d "venv" ]; then
    VENV_DIR="venv"
elif [ -d ".venv" ]; then
    VENV_DIR=".venv"
else
    # Default to .venv for new installations
    VENV_DIR=".venv"
fi

command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
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

if command_exists pip3; then
    PIP=pip3
elif command_exists pip; then
    PIP=pip
else
    echo "pip is not installed. Exiting."
    exit 1
fi

echo "Removing existing '$VENV_DIR'..."
rm -rf $VENV_DIR

echo "Creating new '$VENV_DIR'..."
$PYTHON -m venv $VENV_DIR

# Activate the new environment to install packages into it.
# Handle Windows and Unix-like activation paths.
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

echo "Upgrading pip..."
$PYTHON -m pip install --upgrade pip

echo "Installing requirements..."
$PIP install -r requirements.txt
echo "Requirements installed."
