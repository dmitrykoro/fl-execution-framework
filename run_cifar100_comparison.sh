#!/bin/bash
# CIFAR-100 comparison study: Bulyan, Multi-Krum, Trimmed Mean, FedNGA
# PID and RFA results are taken directly from prior experiment CSVs.

set -e

PYTHON=/Users/ramanzatsarenko/fl-execution-framework-dev/venv/bin/python3
STUDY_DIR=config/simulation_strategies/study/cifar100_comparison
LOG_DIR=logs/cifar100_comparison
mkdir -p "$LOG_DIR"

CONFIGS=(
  "$STUDY_DIR/a01_label_flipping.json"
  "$STUDY_DIR/a05_label_flipping.json"
  "$STUDY_DIR/a20_label_flipping.json"
  "$STUDY_DIR/a01_gradient_scaling.json"
  "$STUDY_DIR/a05_gradient_scaling.json"
  "$STUDY_DIR/a20_gradient_scaling.json"
)

TOTAL=${#CONFIGS[@]}
DONE=0

for cfg in "${CONFIGS[@]}"; do
  DONE=$((DONE + 1))
  name=$(basename "$cfg" .json)
  log="$LOG_DIR/${name}.log"
  echo "[$DONE/$TOTAL] Running: $cfg"
  echo "  Log: $log"
  $PYTHON -m src.simulation_runner "$cfg" > "$log" 2>&1
  if [ $? -eq 0 ]; then
    echo "  DONE"
  else
    echo "  FAILED (check $log)"
  fi
done

echo ""
echo "All $TOTAL runs complete."
