#!/bin/bash
# Stop the data pipeline.

set -euo pipefail

PID_FILE="/tmp/chess-pipeline.pid"
LOG_FILE="/tmp/chess-pipeline.log"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Pipeline may not be running."
    # Try to find it anyway
    PID=$(pgrep -f "prepare_datasets.py" | head -1 || true)
    if [[ -n "$PID" ]]; then
        echo "Found pipeline process: PID $PID"
        kill "$PID" 2>/dev/null && echo "Stopped." || echo "Failed to stop."
    fi
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping pipeline (PID $PID)..."
    kill "$PID"
    # Wait a moment for graceful shutdown
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    echo "Stopped."
else
    echo "Pipeline not running (stale PID $PID)."
fi

rm -f "$PID_FILE"
echo "Last log: tail -20 $LOG_FILE"
