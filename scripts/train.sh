#!/usr/bin/env bash
# Launch SFT training with torchrun across 2 GPUs.
#
# Usage:
#   ./scripts/train.sh                          # default config
#   ./scripts/train.sh --config training/configs/qwen3_30b.yaml
#   ./scripts/train.sh --epochs 1 --output checkpoints/debug
#   ./scripts/train.sh --deepspeed              # enable ZeRO-2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default values
CONFIG="training/configs/qwen3_30b.yaml"
NPROC=2
EXTRA_ARGS=()
USE_DEEPSPEED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --nproc)     NPROC="$2"; shift 2 ;;
    --deepspeed) USE_DEEPSPEED=true; shift ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# Enable DeepSpeed ZeRO-2 by patching the config temporarily
if [[ "$USE_DEEPSPEED" == "true" ]]; then
  echo "[train.sh] DeepSpeed ZeRO-2 enabled"
  EXTRA_ARGS+=("--config" "$CONFIG")
  # Pass deepspeed flag via env so train.py can pick it up
  export DEEPSPEED_ENABLED=1
fi

echo "[train.sh] Config : $CONFIG"
echo "[train.sh] Devices: $NPROC GPUs"
echo ""

torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  training/train.py \
  --config "$CONFIG" \
  "${EXTRA_ARGS[@]}"
