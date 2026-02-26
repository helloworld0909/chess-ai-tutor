"""SFT fine-tuning script for Chess Tutor.

Supports QLoRA fine-tuning of Qwen3.5-35B-A3B (or similar) with 4-bit/8-bit
quantization via bitsandbytes. Designed for torchrun 2-GPU launch.

Usage:
    torchrun --nproc_per_node=2 training/train.py --config training/configs/qwen3.5_35b.yaml
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from datasets import Dataset

# ── caching_allocator_warmup monkey-patch ────────────────────────────────────
# transformers 5.0+ runs caching_allocator_warmup() which pre-allocates up to
# half the GPU VRAM before the quantized model has been created. With QLoRA
# (4-bit) on 32 GiB GPUs this causes an OOM before any training weights are
# ever loaded. Replacing with a no-op is safe — the CUDA caching allocator
# will grow on-demand instead.
try:
    import transformers.modeling_utils as _mu

    def _noop_warmup(*args, **kwargs):  # noqa: D401
        pass

    _mu.caching_allocator_warmup = _noop_warmup
    logger_warmup = logging.getLogger(__name__).getChild("patch")
    logger_warmup.info("caching_allocator_warmup patched to no-op (OOM prevention)")

    import transformers.core_model_loading as _cml

    _cml.GLOBAL_WORKERS = 1
    logger_warmup.info("GLOBAL_WORKERS patched to 1 (serialized loading for OOM prevention)")
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model and LoRA configuration."""

    model_name: str = field(
        default="Qwen/Qwen3.5-35B-A3B",
        metadata={"help": "HuggingFace model name or local path"},
    )
    quantization: str = field(
        default="4bit",
        metadata={"help": "Quantization: 4bit, 8bit, or none"},
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha (2×r)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    """Arguments for data loading."""

    train_file: str = field(
        default="data/processed/train.jsonl",
        metadata={"help": "Training JSONL file"},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation JSONL file"},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length (tokens)"},
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Pack short sequences to fill max_seq_length"},
    )


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> Dataset:
    """Load JSONL file, keeping only samples with source='textbook'."""
    data = []
    total_found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                total_found += 1
                metadata = sample.get("metadata", {})
                if metadata.get("source") == "textbook":
                    if "messages" in sample:
                        data.append({"messages": sample["messages"]})
            except json.JSONDecodeError:
                continue
    logger.info(
        "Loaded %d textbook samples (out of %d total) from %s", len(data), total_found, path
    )
    return Dataset.from_list(data)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think_from_target(messages: list[dict]) -> list[dict]:
    """Remove <think>...</think> blocks from assistant message targets.

    Why: pipeline-generated thinking blocks contain meta-reasoning (notation
    checks, legality loops) that SFT would faithfully imitate. We want SFT to
    teach *what a good coaching comment looks like*, not *how the pipeline
    model happened to think*. Thinking quality is left for GRPO to teach via
    verifiable rewards.

    The model still sees thinking blocks that appear earlier in multi-turn
    conversations (as context), but is not trained to reproduce them.
    """
    result = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = _THINK_RE.sub("", msg.get("content") or "").strip()
            result.append({**msg, "content": content})
        else:
            result.append(msg)
    return result


def format_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Apply the model's native chat template to every sample.

    Uses the tokenizer's apply_chat_template so the format is always
    correct for the model being trained. <think> blocks are stripped from
    assistant targets before formatting (see strip_think_from_target).
    Samples that fail chat template application are filtered out.
    """

    def _fmt(example: dict) -> dict:
        try:
            messages = strip_think_from_target(example["messages"])
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text, "_ok": True}
        except Exception as exc:
            logger.debug("Skipping sample, apply_chat_template failed: %s", exc)
            return {"text": "", "_ok": False}

    dataset = dataset.map(_fmt, remove_columns=["messages"], desc="Applying chat template")
    before = len(dataset)
    dataset = dataset.filter(lambda x: x["_ok"])
    dataset = dataset.remove_columns(["_ok"])
    skipped = before - len(dataset)
    if skipped:
        logger.warning("Filtered %d / %d samples that failed chat template", skipped, before)
    return dataset


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Load model with QLoRA and tokenizer.

    Uses device_map='auto' to split the model across all available GPUs
    within a single process. This is model-parallel (pipeline-style) rather
    than DDP — so start_train.sh must use --nproc_per_node=1.

    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for SFT with packing

    # --- Quantization ---
    bnb_config = None
    if model_args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # --- Model ---
    # Detect DDP (Distributed Data Parallel) from environment
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        logger.info(
            "Distributed training detected (LOCAL_RANK=%d). Disabling device_map.", local_rank
        )
        device_map = None  # Trainer will handle device placement
    else:
        # Single-process logic (supports device_map for multiple GPUs if not in DDP)
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        n_layers = text_cfg.num_hidden_layers

        # Heuristic for POC vs Production scale
        is_small_model = getattr(text_cfg, "hidden_size", 4096) < 4096 or n_layers < 30

        if is_small_model:
            logger.info("Small model detected (%d layers). Using device_map='auto'.", n_layers)
            device_map = "auto"
        else:
            split = n_layers // 2
            device_map = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
            for i in range(n_layers):
                device_map[f"model.layers.{i}"] = 0 if i < split else 1
            logger.info(
                "Large model detected (%d layers). Using manual split: %d/%d.",
                n_layers,
                split,
                n_layers - split,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation="sdpa",
    )

    # Required when using gradient checkpointing with PEFT
    model.enable_input_require_grads()

    # --- LoRA ---
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def build_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    data_args: DataArguments,
    training_args,
):
    """Construct SFTTrainer with response-only masking.

    DataCollatorForCompletionOnlyLM finds every occurrence of the assistant
    response start token sequence and masks everything before it from the loss.
    This ensures the model only learns to predict assistant outputs, not to
    reproduce system/user prompts.
    """
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    # Qwen3 uses ChatML: <|im_start|>assistant\n marks the start of every
    # assistant turn. The collator searches for this token sequence and
    # zeroes out labels for all preceding tokens.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=data_args.max_seq_length,
        packing=data_args.packing,
    )
    return trainer


def train(config_path: str, resume_from_checkpoint: str | bool | None = None):
    """Run SFT training from a YAML config file."""
    from transformers import TrainingArguments

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    model_args = ModelArguments(
        model_name=model_cfg.get("model_name", model_cfg.get("name", "Qwen/Qwen3.5-35B-A3B")),
        quantization=model_cfg.get("quantization", "4bit"),
        lora_r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
    )
    data_args = DataArguments(
        train_file=train_cfg.get("train_file", "data/processed/train.jsonl"),
        eval_file=train_cfg.get("eval_file"),
        max_seq_length=train_cfg.get("max_seq_length", 8192),
        packing=train_cfg.get("packing", True),
    )

    deepspeed_cfg = config.get("deepspeed", {})
    deepspeed_path = deepspeed_cfg.get("config_file") if deepspeed_cfg.get("enabled") else None

    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "checkpoints/chess-tutor-v1"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to="wandb" if wandb_cfg.get("enabled") else "none",
        deepspeed=deepspeed_path,
        use_liger_kernel=True,
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
    )

    # --- Wandb ---
    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
        )

    # --- Data ---
    logger.info("Loading training data from %s", data_args.train_file)
    model, tokenizer = setup_model_and_tokenizer(model_args)

    train_dataset = format_dataset(load_jsonl(data_args.train_file), tokenizer)
    eval_dataset = None
    if data_args.eval_file:
        logger.info("Loading eval data from %s", data_args.eval_file)
        eval_dataset = format_dataset(load_jsonl(data_args.eval_file), tokenizer)

    # --- Train ---
    trainer = build_trainer(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_args,
        training_args,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # --- Save ---
    logger.info("Saving model to %s", training_args.output_dir)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Done.")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT training for Chess Tutor")
    parser.add_argument(
        "--config",
        "-c",
        default="training/configs/qwen3.5_35b.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--data", help="Override train_file from config")
    parser.add_argument("--output", help="Override output_dir from config")
    parser.add_argument("--epochs", type=int, help="Override num_train_epochs")
    parser.add_argument("--lr", type=float, help="Override learning_rate")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume from checkpoint. Pass a path, or omit a value to auto-resume from latest.",
    )

    args = parser.parse_args()
    config_path = args.config

    if any([args.data, args.output, args.epochs, args.lr]):
        # Apply CLI overrides by writing a temp config
        import tempfile

        cfg = load_config(config_path)
        if args.data:
            cfg.setdefault("training", {})["train_file"] = args.data
        if args.output:
            cfg["output_dir"] = args.output
        if args.epochs:
            cfg.setdefault("training", {})["num_train_epochs"] = args.epochs
        if args.lr:
            cfg.setdefault("training", {})["learning_rate"] = args.lr

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            config_path = f.name

    train(config_path, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
