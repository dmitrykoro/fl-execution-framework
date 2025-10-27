#!/bin/bash

set -eu

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/common.sh"

# Navigate to project root and setup environment
navigate_to_root
export PYTHONPATH="$(pwd)"

# Setup logging
setup_logging_with_file "tests/logs" "examples_run"

# Find Python interpreter
find_python_interpreter

log_info "Starting sequential run of all example configs..."
log_and_tee "=================================================="
log_and_tee ""

# Auto-discover all example configs
CONFIGS=()
EXAMPLES_DIR="config/simulation_strategies/examples"
while IFS= read -r -d '' config_path; do
    config_name=$(basename "$config_path")
    CONFIGS+=("$config_name")
done < <(find "$EXAMPLES_DIR" -maxdepth 1 -name "*.json" -type f -print0 | sort -z)

TOTAL="${#CONFIGS[@]}"

# Run each config
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    current=$((i + 1))

    log_and_tee "[$current/$TOTAL] Running $config..."
    if run_python src/simulation_runner.py "examples/$config" --log-level ERROR 2>&1 | tee -a "$LOG_FILE"; then
        log_info "Completed $current/$TOTAL"
    else
        log_error "Failed on $config"
        exit 1
    fi
    log_and_tee ""
done

log_and_tee "=================================================="
log_info "All configs completed!"
log_and_tee ""
log_info "Output directories:"
ls -d out/*/ 2>/dev/null | tail -11 | tee -a "$LOG_FILE" || log_warning "No output directories found"
