#!/bin/bash
# CIFAR-100 Dirichlet non-IID study runner (sequential)
# Alphas: 0.1 (high non-IID), 0.5 (moderate), 2.0 (mild)
# Attacks: no_attack (baseline), label_flipping, gradient_scaling

set -e

PYTHON=/Users/ramanzatsarenko/fl-execution-framework-dev/venv/bin/python3
STUDY_DIR=config/simulation_strategies/study/cifar100
LOG_DIR=logs/cifar100_study
mkdir -p "$LOG_DIR"

CONFIGS=(
  # Baseline (no attack) — each alpha, PID only
  "$STUDY_DIR/a01_no_attack.json"
  "$STUDY_DIR/a05_no_attack.json"
  "$STUDY_DIR/a20_no_attack.json"
  # Label flipping — all strategies
  "$STUDY_DIR/a01_label_flipping.json"
  "$STUDY_DIR/a05_label_flipping.json"
  "$STUDY_DIR/a20_label_flipping.json"
  # Gradient scaling — all strategies
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
