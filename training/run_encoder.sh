#!/usr/bin/env bash
#
# Helper script to launch the Qwen3-4B + ChessEncoder DDP training locally.
#

set -euo pipefail

# Find project root (assumes scripts are in training/ or recipes-train/)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${PROJ_ROOT}"

echo "Config : ${PROJ_ROOT}/training/configs/qwen3_4b_encoder_sft.yaml"
echo "Devices: 2 GPUs (DDP)"
echo "Log    : /tmp/chess-encoder-train.log"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

nohup uv run torchrun \
    --nproc_per_node=2 \
    training/train_encoder.py \
    --config training/configs/qwen3_4b_encoder_sft.yaml \
    > /tmp/chess-encoder-train.log 2>&1 &

PID=$!
echo "Started (PID $PID)"
echo "Monitor: tail -f /tmp/chess-encoder-train.log"
echo "Stop   : pkill -f train_encoder.py"
