#!/usr/bin/env bash
# GRPO training — Qwen3-4B-Thinking-2507 + QLoRA (8-bit, r=32)
# Single GPU (cuda:0) — GRPO rollout generation dominates; DDP adds overhead
# Rewards: R1 legality + R3 eval accuracy + R4a annotations + R5 relevance
# Output: checkpoints/chess-tutor-4b-grpo/
#
# Usage:
#   ./recipes-train/qwen3-4b-phase3-grpo/start.sh
#   ./recipes-train/qwen3-4b-phase3-grpo/start.sh --resume           # resume last checkpoint
#   ./recipes-train/qwen3-4b-phase3-grpo/start.sh --resume checkpoints/chess-tutor-4b-grpo/checkpoint-300
#
# Logs: /tmp/chess-grpo.log
# Stop: ./recipes-train/qwen3-4b-phase3-grpo/stop.sh

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
cd "$REPO_ROOT"

LOG_FILE="/tmp/chess-grpo.log"
PID_FILE="/tmp/chess-grpo.pid"
CONFIG="$RECIPE_DIR/config.yaml"
EXTRA_ARGS=()

# Fail if already running
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "GRPO training already running (PID $EXISTING_PID). Run stop.sh first."
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
export CUDA_VISIBLE_DEVICES=0        # single GPU for GRPO
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
# src/ must be on PYTHONPATH so `from verification.rewards import ...` works
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# Forward all arguments to train.py
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

echo "Config    : $CONFIG"
echo "Device    : cuda:0 (single GPU)"
echo "Stockfish : $STOCKFISH_PATH"
echo "Log       : $LOG_FILE"
echo ""

TRAIN_CMD="python $RECIPE_DIR/train.py --config $CONFIG ${EXTRA_ARGS[*]:-}"

# shellcheck disable=SC2086
nohup bash -c "source $REPO_ROOT/.venv/bin/activate \
  && export PYTORCH_ALLOC_CONF=expandable_segments:True \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export STOCKFISH_PATH=${STOCKFISH_PATH} \
  && export PYTHONPATH=$REPO_ROOT/src:${PYTHONPATH:-} \
  && $TRAIN_CMD" \
  > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

echo "Started (PID $TRAIN_PID)"
echo "Monitor: tail -f $LOG_FILE"
echo "Stop   : $RECIPE_DIR/stop.sh"
