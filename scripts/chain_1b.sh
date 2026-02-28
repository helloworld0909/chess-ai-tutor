#!/bin/bash
# scripts/chain_1b.sh
# 1. Wait for current training to finish
# 2. Generate 1B dataset
# 3. Increase model training parameters (epochs, lr)
# 4. Start 1B training

# --- CONFIG ---
LIMIT=1000000000
WORKERS=16
OUTPUT_TRAIN="data/processed/encoder_pretrain_1b.jsonl"
OUTPUT_EVAL="data/processed/encoder_pretrain_eval_1b.jsonl"
RECIPE_DIR="recipes-train/encoder-pretrain"
PID_FILE="/tmp/encoder-pretrain.pid"

# --- 1. WAIT ---
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "$(date): Waiting for current training (PID $PID) to finish..."
    while kill -0 $PID 2>/dev/null; do
        sleep 300 # check every 5 mins
    done
    echo "$(date): Current training finished."
else
    echo "$(date): No PID file found in $PID_FILE. Skipping wait..."
fi

# --- 2. GENERATE 1B DATA ---
echo "$(date): Starting 1B data generation..."
uv run python data/pipeline/generate_encoder_data.py \
    --output "$OUTPUT_TRAIN" \
    --eval-output "$OUTPUT_EVAL" \
    --limit "$LIMIT" \
    --workers "$WORKERS" \
    --min-ply 4 \
    --max-ply 80 \
    --eval-ratio 0.001

if [ $? -ne 0 ]; then
    echo "$(date): Data generation failed! Aborting."
    exit 1
fi

# --- 3. UPDATE CONFIG ---
echo "$(date): Updating config.yaml for 1B training..."
# Note: we use 'sed' to update the config dynamically for the next run
sed -i "s|train_file: .*|train_file: $OUTPUT_TRAIN|" "$RECIPE_DIR/config.yaml"
sed -i "s|eval_file: .*|eval_file: $OUTPUT_EVAL|" "$RECIPE_DIR/config.yaml"
sed -i "s|num_train_epochs: .*|num_train_epochs: 3|" "$RECIPE_DIR/config.yaml"
sed -i "s|learning_rate: .*|learning_rate: 5.0e-4|" "$RECIPE_DIR/config.yaml"
sed -i "s|eval_steps: .*|eval_steps: 20000|" "$RECIPE_DIR/config.yaml"
sed -i "s|save_steps: .*|save_steps: 20000|" "$RECIPE_DIR/config.yaml"

# --- 4. START TRAINING ---
echo "$(date): Starting 1B training run..."
"$RECIPE_DIR/start.sh"
echo "$(date): Training started with nohup. Monitor /tmp/encoder-pretrain.log"
