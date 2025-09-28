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

    log_info "🔍 Searching for a compatible Python interpreter..."
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

    # Windows timing fix: venv creation may not be atomic
    if [ ! -d "venv/Scripts" ] && [ ! -d "venv/bin" ] && [ -d "venv" ]; then
        sleep 1
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
            # Update PYTHON_CMD to use venv python on Windows
            if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
                PYTHON_CMD="$VENV_DIR/Scripts/python.exe"
                export PYTHON_CMD
            fi
        else
            source "$VENV_DIR/bin/activate"
            # Update PYTHON_CMD to use venv python on Unix
            if [ -f "$VENV_DIR/bin/python" ]; then
                PYTHON_CMD="$VENV_DIR/bin/python"
                export PYTHON_CMD
            fi
        fi
        log_info "Virtual environment activated."
    else
        log_warning "No virtual environment found (expected 'venv' or '.venv')."
    fi
}

get_venv_name() {
    if [ -d ".venv" ]; then
        echo ".venv"
    else
        echo "venv"
    fi
}

ensure_virtual_environment() {
    if [[ -z "${VIRTUAL_ENV:-}" ]] && ! [ -d "venv" ] && ! [ -d ".venv" ]; then
        log_warning "Virtual environment not found. You may need to run './reinstall_requirements.sh' to create one."
        return 1
    fi
    setup_virtual_environment
}

setup_logging_with_file() {
    local log_dir="${1:-tests/logs}"
    local log_prefix="${2:-script}"

    mkdir -p "$log_dir"
    local log_file="$log_dir/${log_prefix}_$(date +%Y%m%d_%H%M%S).log"
    log_info "📝 Logging to: $log_file"
    exec > >(tee "$log_file") 2>&1

    # Export for other functions to use
    export LOG_FILE="$log_file"
    export LOG_DIR="$log_dir"
}

install_requirements() {
    local requirements_file="${1:-requirements.txt}"
    if [ -f "$requirements_file" ]; then
        log_info "📦 Installing requirements from $requirements_file..."
        if pip install -r "$requirements_file"; then
            log_info "Requirements installed successfully."
        else
            log_error "Failed to install requirements from $requirements_file"
            return 1
        fi
    else
        log_warning "Requirements file $requirements_file not found, skipping installation."
    fi
}

check_and_install_tool() {
    local tool_name="$1"
    local install_command="$2"
    local check_command="${3:-command_exists}"

    if ! $check_command "$tool_name"; then
        log_warning "$tool_name not found. Attempting to install..."
        if eval "$install_command"; then
            log_info "$tool_name installed successfully."
        else
            log_error "Failed to install $tool_name. Please install manually."
            return 1
        fi
    fi
}

show_simulation_output_info() {
    local output_dir="${1:-out/}"

    # Convert to relative path
    local rel_output_dir="${output_dir#$(pwd)/}"

    # Find the most recent timestamped directory
    local latest_dir=""
    if [ -d "$output_dir" ]; then
        latest_dir=$(find "$output_dir" -maxdepth 1 -type d -name "*-*-*_*-*-*" | sort | tail -1)
        if [ -n "$latest_dir" ]; then
            latest_dir="${latest_dir#$(pwd)/}"
        fi
    fi

    log_info "🎉 Simulation completed successfully!"

    if [ -n "$latest_dir" ]; then
        log_info "📁 Results saved to: $latest_dir"
        log_info "📊 Contains: plots, CSV data, configs, datasets, logs"
    else
        log_info "📁 Results saved to: $rel_output_dir"
    fi
}

# Environment setup utilities
setup_unicode_env() {
    export PYTHONIOENCODING="utf-8"
    log_info "Unicode environment configured (PYTHONIOENCODING=utf-8)"
}

run_python_with_unicode() {
    local script_path="$1"
    shift  # Remove first argument, keep the rest

    setup_unicode_env

    if [[ -n "${VIRTUAL_ENV:-}" ]] || [ -d "venv" ] || [ -d ".venv" ]; then
        setup_virtual_environment
    fi

    if [[ -z "${PYTHON_CMD:-}" ]]; then
        find_python_interpreter
    fi

    # Run with proper environment
    PYTHONIOENCODING=utf-8 $PYTHON_CMD "$script_path" "$@"
}

run_pytest_with_unicode() {
    local test_path="${1:-.}"
    shift

    setup_unicode_env

    if [[ -n "${VIRTUAL_ENV:-}" ]] || [ -d "venv" ] || [ -d ".venv" ]; then
        setup_virtual_environment
    fi

    if [[ -z "${PYTHON_CMD:-}" ]]; then
        find_python_interpreter
    fi

    # Run pytest with proper environment
    PYTHONIOENCODING=utf-8 $PYTHON_CMD -m pytest "$test_path" "$@"
}

PYTHON_CMD=""
export PYTHON_CMD
