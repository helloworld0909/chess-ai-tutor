#!/bin/bash
# Stop vLLM server for Chess AI Tutor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/docker"

echo "Stopping vLLM server..."
docker compose down

echo "vLLM stopped."
