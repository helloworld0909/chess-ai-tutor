#!/usr/bin/env bash
# Run the test suite with the correct environment.
#
# Usage:
#   ./scripts/test.sh              # run all tests
#   ./scripts/test.sh tests/test_web.py        # specific file
#   ./scripts/test.sh -k "chesscom"            # filter by name
#   ./scripts/test.sh -x                       # stop on first failure
#   ./scripts/test.sh -v                       # verbose output

set -euo pipefail

STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export STOCKFISH_PATH

cd "$REPO_ROOT"

exec uv run pytest tests/ "$@"
