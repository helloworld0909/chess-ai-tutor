#!/bin/bash
# Stop vLLM server — Qwen3.5-35B-A3B-FP8

set -e
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping Qwen3.5-35B-A3B-FP8 vLLM server..."
docker compose -f "$RECIPE_DIR/docker-compose.yml" down
echo "Done."
