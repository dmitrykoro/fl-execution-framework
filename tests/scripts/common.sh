#!/bin/sh
# Shell utilities for FL execution framework

set -eu

# ============================================================================
# Logging
# ============================================================================

log_info() {
    echo "✅ $1"
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1" >&2
}

setup_logging_with_file() {
    _log_dir="${1:-tests/logs}"
    _log_prefix="${2:-script}"

    mkdir -p "$_log_dir"
    LOG_FILE="$_log_dir/${_log_prefix}_$(date +%Y%m%d_%H%M%S).log"
    : > "$LOG_FILE"

    log_info "📝 Logging to: $LOG_FILE"

    export LOG_FILE
    LOG_DIR="$_log_dir"
    export LOG_DIR
}

log_and_tee() {
    if [ -n "${LOG_FILE:-}" ]; then
        printf "%s\n" "$*" | tee -a "$LOG_FILE"
    else
        printf "%s\n" "$*"
    fi
}

# ============================================================================
# System
# ============================================================================

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

navigate_to_root() {
    if [ -f "requirements.txt" ] && [ -d "src" ]; then
        return
    fi
    script_dir="$(cd "$(dirname "$0")" && pwd)"

    case "$script_dir" in
        *"/tests/scripts")
            cd "$script_dir/../.."
            ;;
        *"/tests")
            cd "$script_dir/.."
            ;;
        *)
            log_warning "Could not determine project root. Staying in $(pwd)."
            ;;
    esac
}

get_physical_cores() {
    if [ -z "${PYTHON_CMD:-}" ]; then
        find_python_interpreter
    fi
    "$PYTHON_CMD" -c "import psutil; print(psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1)"
}

# ============================================================================
# Python Environment
# ============================================================================

find_python_interpreter() {
    if [ -n "${PYTHON_CMD:-}" ] && command_exists "$PYTHON_CMD"; then
        return 0
    fi

    log_info "🔍 Searching for a compatible Python interpreter..."
    for version in python3.11 python3.10 python3.9 python3 python; do
        if command_exists "$version"; then
            if "$version" -c "import sys; sys.exit(not (sys.version_info >= (3, 9) and sys.version_info < (3, 12)))" 2>/dev/null; then
                PYTHON_CMD="$version"
                export PYTHON_CMD
                log_info "Found compatible Python: $PYTHON_CMD"
                return 0
            fi
        fi
    done

    if command_exists py; then
        for py_version in "-3.11" "-3.10" "-3.9" "-3"; do
             if py "$py_version" -c "import sys; sys.exit(not (sys.version_info >= (3, 9) and sys.version_info < (3, 12)))" 2>/dev/null; then
                PYTHON_CMD="py $py_version"
                export PYTHON_CMD
                log_info "Found compatible Python via 'py' launcher: $PYTHON_CMD"
                return 0
            fi
        done
    fi

    log_error "Python 3.9, 3.10, or 3.11 was not found. Please install a compatible version."
    exit 1
}

setup_virtual_environment() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        VENV_DIR="$VIRTUAL_ENV"
        export VENV_DIR
        return 0
    fi

    # Windows: venv creation may not be atomic
    if [ ! -d "venv/Scripts" ] && [ ! -d "venv/bin" ] && [ -d "venv" ]; then
        sleep 1
    fi

    log_info "🔌 Searching for virtual environment..."
    venv_path=""
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
            . "$VENV_DIR/Scripts/activate"
            if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
                PYTHON_CMD="$VENV_DIR/Scripts/python.exe"
                export PYTHON_CMD
            fi
        else
            . "$VENV_DIR/bin/activate"
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
    if [ -z "${VIRTUAL_ENV:-}" ] && ! [ -d "venv" ] && ! [ -d ".venv" ]; then
        log_warning "Virtual environment not found. You may need to run './reinstall_requirements.sh' to create one."
        return 1
    fi
    setup_virtual_environment
}

setup_unicode_env() {
    export PYTHONIOENCODING="utf-8"
    log_info "Unicode environment configured (PYTHONIOENCODING=utf-8)"
}

setup_joblib_env() {
    LOKY_MAX_CPU_COUNT=$(get_physical_cores)
    export LOKY_MAX_CPU_COUNT
    export KMP_DUPLICATE_LIB_OK=TRUE
    export PYTHONWARNINGS="ignore::RuntimeWarning:threadpoolctl"
}

# ============================================================================
# Dependencies
# ============================================================================

install_requirements() {
    requirements_file="${1:-requirements.txt}"
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
    tool_name="$1"
    install_command="$2"
    check_command="${3:-command_exists}"

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

# ============================================================================
# Python Execution
# ============================================================================

run_python_with_unicode() {
    script_path="$1"
    shift

    setup_unicode_env

    if [ -n "${VIRTUAL_ENV:-}" ] || [ -d "venv" ] || [ -d ".venv" ]; then
        setup_virtual_environment
    fi

    if [ -z "${PYTHON_CMD:-}" ]; then
        find_python_interpreter
    fi

    PYTHONIOENCODING=utf-8 $PYTHON_CMD "$script_path" "$@"
}

run_pytest_with_unicode() {
    test_path="${1:-.}"
    shift

    setup_unicode_env

    if [ -n "${VIRTUAL_ENV:-}" ] || [ -d "venv" ] || [ -d ".venv" ]; then
        setup_virtual_environment
    fi

    if [ -z "${PYTHON_CMD:-}" ]; then
        find_python_interpreter
    fi

    PYTHONIOENCODING=utf-8 $PYTHON_CMD -m pytest "$test_path" "$@"
}

# ============================================================================
# Simulation
# ============================================================================

show_simulation_output_info() {
    output_dir="${1:-out/}"
    _pwd="$(pwd)"

    rel_output_dir="${output_dir#"$_pwd"/}"

    latest_dir=""
    if [ -d "$output_dir" ]; then
        latest_dir=$(find "$output_dir" -maxdepth 1 -type d -name "*-*-*_*-*-*" | sort | tail -1)
        if [ -n "$latest_dir" ]; then
            latest_dir="${latest_dir#"$_pwd"/}"
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

# ============================================================================
# Globals
# ============================================================================

PYTHON_CMD=""
export PYTHON_CMD
