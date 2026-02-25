#!/bin/bash
# Start the data pipeline (LLM coaching generation).
#
# Usage:
#   ./scripts/start_pipeline.sh                    # default
#   ./scripts/start_pipeline.sh --only-sources textbook
#   ./scripts/start_pipeline.sh --llm-workers 64

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/chess-pipeline.log"
PID_FILE="/tmp/chess-pipeline.pid"

# Check if already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Pipeline is already running (PID $(cat "$PID_FILE"))."
    echo "Use './scripts/stop_pipeline.sh' to stop it first."
    echo "Logs: tail -f $LOG_FILE"
    exit 0
fi

# Default arguments
STOCKFISH="/home/zheng/.local/bin/stockfish"
LLM_MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
TEXTBOOK_PATH="data/raw/textbook_augmented.jsonl"
EXTRA_ARGS=("$@")

cd "$PROJECT_DIR"

echo "Starting data pipeline..."
echo "  Model:   $LLM_MODEL"
echo "  Log:     $LOG_FILE"
echo ""

nohup uv run python data/pipeline/prepare_datasets.py \
  --stockfish "$STOCKFISH" \
  --output-dir data/processed \
  --max-per-source 0 \
  --workers 16 \
  --llm-coach \
  --llm-model "$LLM_MODEL" \
  --textbook-path "$TEXTBOOK_PATH" \
  --llm-workers 128 \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Pipeline started (PID $(cat "$PID_FILE"))."
echo ""
echo "  View logs:  tail -f $LOG_FILE"
echo "  Stop:       ./scripts/stop_pipeline.sh"
