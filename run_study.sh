#!/bin/bash
# Comprehensive experimental study: IID vs NIID robustness
# Runs all 9 configs × 3 seeds sequentially, logging to /tmp/study_*.log

PYTHON=/Users/ramanzatsarenko/fl-execution-framework-dev/venv/bin/python3
export LOKY_MAX_CPU_COUNT=4
export KMP_DUPLICATE_LIB_OK=TRUE

CONFIGS=(
    "config/simulation_strategies/study/iid_50c_label_flipping.json"
    "config/simulation_strategies/study/iid_50c_gaussian_noise.json"
    "config/simulation_strategies/study/iid_50c_gradient_scaling.json"
    "config/simulation_strategies/study/iid_12c_label_flipping.json"
    "config/simulation_strategies/study/iid_12c_gaussian_noise.json"
    "config/simulation_strategies/study/iid_12c_gradient_scaling.json"
    "config/simulation_strategies/study/niid_12c_label_flipping.json"
    "config/simulation_strategies/study/niid_12c_gaussian_noise.json"
    "config/simulation_strategies/study/niid_12c_gradient_scaling.json"
)

SEEDS=3
TOTAL=$(( ${#CONFIGS[@]} * SEEDS ))
RUN=0

echo "=== Starting study: ${#CONFIGS[@]} configs × ${SEEDS} seeds = ${TOTAL} runs ===" | tee /tmp/study_progress.log

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .json)
    for seed in $(seq 1 $SEEDS); do
        RUN=$((RUN + 1))
        log="/tmp/study_${name}_seed${seed}.log"
        echo "[${RUN}/${TOTAL}] $(date '+%H:%M:%S') — $name seed $seed" | tee -a /tmp/study_progress.log
        $PYTHON -m src.simulation_runner "$cfg" > "$log" 2>&1
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ done" | tee -a /tmp/study_progress.log
        else
            echo "  ✗ FAILED (exit $exit_code) — see $log" | tee -a /tmp/study_progress.log
        fi
        # brief pause between runs to let Ray fully clean up
        sleep 3
    done
done

echo "=== Study complete $(date) ===" | tee -a /tmp/study_progress.log
