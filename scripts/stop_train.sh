#!/usr/bin/env bash
# Gracefully stop a training run started by scripts/train.sh.
#
# Usage:
#   ./scripts/stop_train.sh

PID_FILE="/tmp/chess-train.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[stop_train.sh] No PID file found ($PID_FILE). Is training running?"
  # Fallback: kill any stray torchrun / train.py processes
  STRAY=$(pgrep -f "train\.py|torchrun" 2>/dev/null || true)
  if [[ -n "$STRAY" ]]; then
    echo "[stop_train.sh] Found stray processes: $STRAY — killing them."
    pkill -SIGTERM -f "train\.py|torchrun" 2>/dev/null || true
  fi
  exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
  echo "[stop_train.sh] Process $PID is not running (already finished or crashed)."
  rm -f "$PID_FILE"
  exit 0
fi

echo "[stop_train.sh] Sending SIGTERM to process group of PID $PID..."
# Kill the whole process group (torchrun + worker ranks)
kill -SIGTERM -- "-$PID" 2>/dev/null || kill -SIGTERM "$PID" 2>/dev/null

# Wait up to 30s for graceful shutdown
for i in $(seq 1 30); do
  if ! kill -0 "$PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

# Force kill if still alive
if kill -0 "$PID" 2>/dev/null; then
  echo "[stop_train.sh] Process still alive after 30s — sending SIGKILL."
  kill -SIGKILL -- "-$PID" 2>/dev/null || kill -SIGKILL "$PID" 2>/dev/null
fi

rm -f "$PID_FILE"
echo "[stop_train.sh] Done."
