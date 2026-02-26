# Chess Tutor SFT Training Summary (4B-Thinking POC)

This document summarizes the development and launch of the Supervised Fine-Tuning (SFT) pipeline for the Chess Tutor project.

## 1. Project Objective
Fine-tune a Qwen-series model to act as a chess coach, utilizing "thinking" (reasoning) capabilities to provide high-quality move analysis and coaching feedback.

## 2. Technical Challenges & Evolutions

### Initial 35B Model Attempt
*   **Model**: `Qwen/Qwen3.5-35B-A3B` (Base/FP8).
*   **Issue**: Persistent `OutOfMemoryError` (OOM) during model materialization on 32GB RTX 5090 GPUs. Even with 4-bit QLoRA and manual layer splitting, the BF16 materialization spikes exceeded the 32GB limit.
*   **Optimization**: Implemented several monkey-patches to `transformers` to disable aggressive pre-allocation (`caching_allocator_warmup`) and serialize shard loading (`GLOBAL_WORKERS = 1`).

### Pivot to 4B Proof-of-Concept (POC)
*   **Model**: `Qwen/Qwen3-4B-Thinking-2507`.
*   **Rationale**: Smaller footprint allows for verifying the full pipeline, data filtering, and reasoning capture without hardware-level blockers.

## 3. Data Engineering
*   **Textbook Filtering**: Implemented logical filtering to use only high-quality coaching samples (source: `textbook`).
    *   **Train Set**: 23,439 samples.
    *   **Eval Set**: 2,640 samples.
*   **Instruction Masking**: Used `DataCollatorForCompletionOnlyLM` with the `<|im_start|>assistant\n` template to ensure the model only learns to generate the assistant's reasoning and coaching, not the prompts.

## 4. Hardware Scaling & Parallelism
*   **Dual GPU Scaling (4B Model)**:
    *   **Strategy**: **Distributed Data Parallel (DDP)** via `torchrun`.
    *   **Execution**: Model is replicated on both GPUs; data is split across them.
    *   **Quantization**: **8-bit** (Int8).
    *   **Efficiency**: High; both GPUs are utilized at 100% compute capacity.
    *   **Memory**: ~17GB VRAM per GPU (weights replicated).

*   **Projection for 35B Model**:
    *   **Constraint**: A 35B model in 4-bit requires ~17.5GB for weights alone. With DDP, adding activations (8k context) would exceed the 32GB limit (~45GB+ required).
    *   **Strategy**: **Pipeline Parallelism (PP)**. Layers will be split (e.g., 40/40) across the two GPUs.
    *   **Trade-off**: PP is significantly slower than DDP (roughly 15x-20x slowdown) but is the only way to fit the model without DeepSpeed ZeRO-3.

## 5. Training Insights & Logic
*   **Loss Function**: Cross-Entropy loss applied only to the assistant's responses (**Completion-Only masking**). The model is not trained on the user's prompts.
*   **Thinking Block Masking**: `<think>...</think>` blocks are stripped from the training data during SFT. 
    *   **Goal**: Prevent imitation of pipeline-generated meta-reasoning. 
    *   **Future**: Reasoning quality is deferred to Reinforcement Learning (GRPO).
*   **Epoch Strategy**: 
    *   **Current**: 3 epochs for the 4B model to ensure style internalization.
    *   **35B Plan**: Recommend 1 epoch due to high intelligence and massive training time (~40+ hours for 3 epochs).

## 6. Current Training Status
*   **Status**: Active & Stable.
*   **Logs**: `/tmp/chess-train.log` (reporting loss settlements around 2.4-2.5).
*   **Checkpoints**: Every 500 steps in `checkpoints/chess-tutor-4b-poc/` (keeping top 3).

## 7. Next Steps
*   **Monitor Eval Loss**: Every 200 steps (as configured in the YAML).
*   **Manual Review**: Use `src/tutor/review.py` to verify coaching quality once first checkpoints are ready.
*   **Go/No-Go for 35B**: Decided based on the coached output quality of the 4B model.
