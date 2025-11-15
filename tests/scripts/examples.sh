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

# Auto-discover all example configs
CONFIGS=()
EXAMPLES_DIR="config/simulation_strategies/testing"
while IFS= read -r -d '' config_path; do
    config_name=$(basename "$config_path")
    CONFIGS+=("$config_name")
done < <(find "$EXAMPLES_DIR" -maxdepth 1 -name "*.json" -type f -print0 | sort -z)

TOTAL="${#CONFIGS[@]}"

# Interactive menu
echo ""
echo "=================================================="
echo "Example Config Runner"
echo "=================================================="
echo ""
echo "Select a config to run:"
echo "  0) Run all configs"
for i in "${!CONFIGS[@]}"; do
    echo "  $((i + 1))) ${CONFIGS[$i]}"
done
echo ""
read -p "Enter selection: " selection

# Validate input
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 0 ] || [ "$selection" -gt "$TOTAL" ]; then
    log_error "Invalid selection"
    exit 1
fi

log_and_tee ""
log_and_tee "=================================================="

# Run selected config(s)
if [ "$selection" -eq 0 ]; then
    log_info "Running all example configs..."
    log_and_tee ""

    for i in "${!CONFIGS[@]}"; do
        config_filename="${CONFIGS[$i]}"
        current=$((i + 1))

        log_and_tee "[$current/$TOTAL] Running $config_filename..."
        run_python src/simulation_runner.py "testing/$config_filename" --log-level DEBUG 2>&1 | tee -a "$LOG_FILE"
        exit_code="${PIPESTATUS[0]}"
        if [ "$exit_code" -eq 0 ]; then
            log_info "Completed $current/$TOTAL"
        else
            log_error "Failed on $config_filename"
            exit 1
        fi
        log_and_tee ""
    done

    log_and_tee "=================================================="
    log_info "All configs completed!"
else
    config_filename="${CONFIGS[$((selection - 1))]}"
    log_info "Running $config_filename..."
    log_and_tee ""

    run_python src/simulation_runner.py "testing/$config_filename" --log-level DEBUG 2>&1 | tee -a "$LOG_FILE"
    exit_code="${PIPESTATUS[0]}"
    if [ "$exit_code" -eq 0 ]; then
        log_info "Completed successfully"
    else
        log_error "Failed on $config_filename"
        exit 1
    fi

    log_and_tee "=================================================="
    log_info "Config completed!"
fi

log_and_tee ""
log_info "Output directories:"
ls -d out/*/ 2>/dev/null | tail -11 | tee -a "$LOG_FILE" || log_warning "No output directories found"
