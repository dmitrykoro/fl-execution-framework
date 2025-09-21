#!/bin/bash

VENV_DIR="venv"

command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

# Check for Python 3.9+ versions in order of preference
PYTHON=""

# Try standard commands first (works on most systems)
for version in python3.11 python3.10 python3.9 python3 python; do
    if command_exists $version; then
        # Check if version is 3.9+
        VERSION_CHECK=$($version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON=$version
            echo "Using Python version: $($version --version)"
            break
        fi
    fi
done

# If no standard commands work, try Windows Python Launcher (py.exe)
if [ -z "$PYTHON" ] && command_exists py; then
    # Try different Python versions with py launcher
    for version in "-3.11" "-3.10" "-3.9" "-3"; do
        VERSION_CHECK=$(py $version -c "import sys; print(sys.version_info >= (3, 9))" 2>/dev/null)
        if [ "$VERSION_CHECK" = "True" ]; then
            PYTHON="py $version"
            echo "Using Python version: $(py $version --version)"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "Python 3.9+ is not installed. Please install Python 3.9 or higher."
    echo "Tried:"
    echo "  - Standard commands: python3.11, python3.10, python3.9, python3, python"
    echo "  - Windows Python Launcher: py -3.11, py -3.10, py -3.9, py -3"
    echo ""
    echo "Install Python from: https://www.python.org/downloads/"
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
# Windows vs Unix activation script paths
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source $VENV_DIR/Scripts/activate
else
    source $VENV_DIR/bin/activate
fi

echo "Installing requirements..."
$PIP install -r requirements.txt
echo "Requirements installed."
