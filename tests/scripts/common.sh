#!/bin/bash
# Common shell utilities

set -euo pipefail
log_info() {
    echo "✅ $1"
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1" >&2
}

navigate_to_root() {
    if [ -f "requirements.txt" ] && [ -d "src" ]; then
        return
    fi
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [[ "$script_dir" == *"/tests/scripts" ]]; then
        cd "$script_dir/../.."
    elif [[ "$script_dir" == *"/tests" ]]; then
        cd "$script_dir/.."
    else
        log_warning "Could not determine project root. Staying in $(pwd)."
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

find_python_interpreter() {
    if [[ -n "${PYTHON_CMD:-}" ]] && command_exists "$PYTHON_CMD"; then
        return 0
    fi

    log_info "🔍 Searching for a Python 3.9+ interpreter..."
    for version in python3.11 python3.10 python3.9 python3 python; do
        if command_exists "$version"; then
            if "$version" -c "import sys; sys.exit(not (sys.version_info >= (3, 9)))" 2>/dev/null; then
                PYTHON_CMD="$version"
                export PYTHON_CMD
                log_info "Found compatible Python: $PYTHON_CMD"
                return 0
            fi
        fi
    done

    if command_exists py; then
        for py_version in "-3.11" "-3.10" "-3.9" "-3"; do
             if py "$py_version" -c "import sys; sys.exit(not (sys.version_info >= (3, 9)))" 2>/dev/null; then
                PYTHON_CMD="py $py_version"
                export PYTHON_CMD
                log_info "Found compatible Python via 'py' launcher: $PYTHON_CMD"
                return 0
            fi
        done
    fi

    log_error "Python 3.9+ was not found. Please install a compatible version."
    exit 1
}

setup_virtual_environment() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        VENV_DIR="$VIRTUAL_ENV"
        export VENV_DIR
        return 0
    fi

    log_info "🔌 Searching for virtual environment..."
    local venv_path=""
    if [ -d "venv" ]; then
        venv_path="venv"
    elif [ -d ".venv" ]; then
        venv_path=".venv"
    fi

    if [ -n "$venv_path" ]; then
        log_info "Found virtual environment in '$venv_path', activating..."
        VENV_DIR="$venv_path"
        export VENV_DIR
        if [ -f "$VENV_DIR/Scripts/activate" ]; then
            source "$VENV_DIR/Scripts/activate"
        else
            source "$VENV_DIR/bin/activate"
        fi
        log_info "Virtual environment activated."
    else
        log_warning "No virtual environment found (expected 'venv' or '.venv')."
    fi
}

PYTHON_CMD=""
export PYTHON_CMD
