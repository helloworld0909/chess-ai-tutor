#!/bin/bash
# Stop the Chess Game Review web UI.

set -euo pipefail

PID_FILE="/tmp/chess-review.pid"
LOG_FILE="/tmp/chess-review.log"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found. Server may not be running."
    PID=$(pgrep -f "chess-review" | head -1 || true)
    if [[ -n "$PID" ]]; then
        echo "Found review process: PID $PID"
        kill "$PID" 2>/dev/null && echo "Stopped." || echo "Failed to stop."
    fi
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping review server (PID $PID)..."
    kill "$PID"
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    echo "Stopped."
else
    echo "Server not running (stale PID $PID)."
fi

rm -f "$PID_FILE"
echo "Last log: tail -20 $LOG_FILE"
