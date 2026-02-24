#!/bin/bash
# Start vLLM server for Chess AI Tutor

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/docker"

echo "Starting vLLM server..."
echo "Model: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
echo "Port: 8100"
echo ""

# Check if already running
if docker ps --format '{{.Names}}' | grep -q '^chess-ai-vllm$'; then
    echo "vLLM is already running."
    echo "Use 'docker logs -f chess-ai-vllm' to view logs"
    exit 0
fi

# Start vLLM
docker compose up -d

echo ""
echo "vLLM starting... (this may take 1-2 minutes to load the model)"
echo ""
echo "Check status:  docker ps | grep chess-ai-vllm"
echo "View logs:     docker logs -f chess-ai-vllm"
echo "Health check:  curl http://localhost:8100/health"
echo ""
echo "Once healthy, run the chess agent:"
echo "  STOCKFISH_PATH=/home/zheng/.local/bin/stockfish \\"
echo "    uv run python src/tutor/agent.py --base-url http://localhost:8100/v1"
