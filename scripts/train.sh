#!/usr/bin/env bash
# Launch SFT training with torchrun across 2 GPUs.
#
# Usage:
#   ./scripts/train.sh                          # default config
#   ./scripts/train.sh --config training/configs/qwen3.5_35b.yaml
#   ./scripts/train.sh --epochs 1 --output checkpoints/debug
#   ./scripts/train.sh --deepspeed              # enable ZeRO-2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate project venv so torchrun / training deps are on PATH
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Avoid OOM from caching_allocator_warmup in transformers 5.x
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Suppress non-actionable advisory warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Default values
CONFIG="training/configs/qwen3.5_35b.yaml"
NPROC=1  # model-parallel (device_map=auto) — single process spans both GPUs
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

# Start training
if [ "$NPROC" -gt 1 ]; then
    echo "[train.sh] Launching with torchrun (DDP) on $NPROC GPUs"
    torchrun --nproc_per_node="$NPROC" training/train.py \
        --config "$CONFIG" \
        "${EXTRA_ARGS[@]}"
else
    echo "[train.sh] Launching with python3 (single process)"
    python3 training/train.py \
        --config "$CONFIG" \
        "${EXTRA_ARGS[@]}"
fi
