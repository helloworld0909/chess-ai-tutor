#!/bin/bash
# Start vLLM server — Qwen3.5-35B-A3B-FP8 (base, no LoRA)
# Port: 8100  |  Tensor parallel: 2  |  Max context: 32768

set -e
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps --format '{{.Names}}' | grep -q '^chess-ai-vllm-35b$'; then
    echo "Already running. Logs: docker logs -f chess-ai-vllm-35b"
    exit 0
fi

echo "Starting Qwen3.5-35B-A3B-FP8 on port 8100..."
docker compose -f "$RECIPE_DIR/docker-compose.yml" up -d

echo ""
echo "Health check : curl http://localhost:8100/health"
echo "View logs    : docker logs -f chess-ai-vllm-35b"
echo "Stop         : $(dirname "$RECIPE_DIR")/recipes/qwen3.5-35b-a3b-fp8/stop.sh"
