#!/bin/bash
# Start the Chess Game Review web UI.
#
# Usage:
#   ./scripts/start-review.sh <username>
#   ./scripts/start-review.sh <username> --port 8080 --months 3
#   ./scripts/start-review.sh <username> --no-browser
#
# Environment variables (or defaults):
#   CHESS_COM_USERNAME  — fallback if no username argument given
#   STOCKFISH_PATH      — path to Stockfish binary
#   LLM_BASE_URL        — vLLM API base URL (default: http://localhost:8100/v1)
#   LLM_MODEL           — model name (default: chess-tutor)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/chess-review.log"
PID_FILE="/tmp/chess-review.pid"

# Load .env if present
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# Resolve username: first positional arg, then env var
USERNAME="${1:-${CHESS_COM_USERNAME:-}}"
if [[ -z "$USERNAME" ]]; then
    echo "Error: no username provided."
    echo "Usage: $0 <username> [options]"
    echo "   or: set CHESS_COM_USERNAME in .env"
    exit 1
fi
# Consume the first arg so remaining args pass through
if [[ $# -ge 1 && "$1" != --* ]]; then
    shift
fi

# Defaults
STOCKFISH="${STOCKFISH_PATH:-/home/zheng/.local/bin/stockfish}"
LLM_URL="${LLM_BASE_URL:-http://localhost:8100/v1}"
LLM_MODEL="${LLM_MODEL:-chess-tutor}"
PORT=8080

# Check if already running
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Review server already running (PID $(cat "$PID_FILE"))."
    echo "  Logs:  tail -f $LOG_FILE"
    echo "  Stop:  ./scripts/stop-review.sh"
    exit 0
fi

cd "$PROJECT_DIR"

echo "Starting Chess Game Review..."
echo "  User:    $USERNAME"
echo "  LLM:     $LLM_URL  ($LLM_MODEL)"
echo "  Log:     $LOG_FILE"
echo ""

nohup env \
    STOCKFISH_PATH="$STOCKFISH" \
    LLM_BASE_URL="$LLM_URL" \
    LLM_MODEL="$LLM_MODEL" \
    uv run chess-review "$USERNAME" \
        --port "$PORT" \
        --no-browser \
        "$@" \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Server started (PID $(cat "$PID_FILE"))."
echo ""
echo "  Open:   http://localhost:$PORT"
echo "  Logs:   tail -f $LOG_FILE"
echo "  Stop:   ./scripts/stop-review.sh"
