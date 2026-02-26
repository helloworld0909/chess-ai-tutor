#!/usr/bin/env bash
# SFT training — Qwen3.5-35B-A3B + QLoRA (4-bit NF4, r=8)
# Single process: device_map=auto spans both GPUs (pipeline parallel)
# Effective batch: 2/GPU × 8 grad-accum × 2 GPUs = 32
# Output: checkpoints/chess-tutor-35b-sft-v1/
#
# Usage:
#   ./recipes-train/qwen3.5-35b-sft/start.sh
#   ./recipes-train/qwen3.5-35b-sft/start.sh --deepspeed
#
# Logs: /tmp/chess-train.log
# Stop: ./recipes-train/qwen3.5-35b-sft/stop.sh

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/chess-train.log"
PID_FILE="/tmp/chess-train.pid"
CONFIG="$RECIPE_DIR/config.yaml"
EXTRA_ARGS=()
USE_DEEPSPEED=false

# Fail if already running
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Training already running (PID $EXISTING_PID). Run stop.sh first."
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Parse extra arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deepspeed) USE_DEEPSPEED=true; shift ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$USE_DEEPSPEED" == "true" ]]; then
  echo "DeepSpeed ZeRO-2 enabled"
  export DEEPSPEED_ENABLED=1
fi

echo "Config : $CONFIG"
echo "Devices: 2 GPUs (pipeline parallel, single process)"
echo "Log    : $LOG_FILE"
echo ""

TRAIN_CMD="python3 $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

# shellcheck disable=SC2086
nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTORCH_ALLOC_CONF=expandable_segments:True \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "Started (PID $TRAIN_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : $RECIPE_DIR/stop.sh"
