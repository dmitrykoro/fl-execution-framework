#!/bin/bash
# PID coefficient ablation study: P-only, PI, PID (low Ki), PID (default), PID (high Ki)
# Label flipping attack, FEMNIST IID, 50 clients, 40 rounds

set -e

PYTHON=python3

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <study_dir>"
  exit 1
fi
# e.g config/simulation_strategies/study/pid_coeff_ablation
STUDY_DIR="$1"
echo "Using study directory: $STUDY_DIR"

LOG_DIR=logs/pid_coeff_ablation
mkdir -p "$LOG_DIR"

CONFIGS=(
  "$STUDY_DIR/p_only.json"
  "$STUDY_DIR/pi.json"
  "$STUDY_DIR/pid_low_ki.json"
  "$STUDY_DIR/pid.json"
  "$STUDY_DIR/pid_high_ki.json"
)

# Check Study Configs before starting
echo "=== Review Study Configs ==="
for cfg in "${CONFIGS[@]}"; do
  echo "  $cfg"
done
echo
read -p "Are these configs correct? (y/n) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborting. Please update the configs and rerun"
  exit 1
fi

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
