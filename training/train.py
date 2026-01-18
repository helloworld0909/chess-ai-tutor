"""Fine-tuning script for Chess Tutor using Unsloth.

Supports LoRA fine-tuning of Qwen3-VL-30B-A3B-Thinking with 8-bit quantization.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    TrainingArguments,
    HfArgumentParser,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name: str = field(
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        metadata={"help": "Model name or path"},
    )
    quantization: str = field(
        default="8bit",
        metadata={"help": "Quantization: 4bit, 8bit, or none"},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Freeze vision tower"},
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    train_file: str = field(
        default="data/processed/train.jsonl",
        metadata={"help": "Training data file"},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation data file"},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(data_path: str) -> Dataset:
    """Load training data from JSONL file.

    Expects format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        ...
    }
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                if "messages" in sample:
                    data.append(sample)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(data)} training samples")
    return Dataset.from_list(data)


def format_messages_for_training(example: dict) -> dict:
    """Format messages into training text.

    Converts conversation format to model input/output format.
    """
    messages = example.get("messages", [])

    # Build conversation text
    text_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            text_parts.append(f"<|system|>\n{content}<|end|>")
        elif role == "user":
            if isinstance(content, list):
                # Multimodal content
                text_content = ""
                for item in content:
                    if item.get("type") == "text":
                        text_content += item.get("text", "")
                text_parts.append(f"<|user|>\n{text_content}<|end|>")
            else:
                text_parts.append(f"<|user|>\n{content}<|end|>")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>\n{content}<|end|>")

    return {"text": "\n".join(text_parts)}


def setup_model_and_tokenizer(
    model_args: ModelArguments,
    use_unsloth: bool = True,
):
    """Set up model and tokenizer with LoRA.

    Args:
        model_args: Model configuration
        use_unsloth: Use Unsloth for faster training

    Returns:
        Tuple of (model, tokenizer)
    """
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            logger.info("Using Unsloth for faster training")

            # Load with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_args.model_name,
                max_seq_length=8192,
                dtype=torch.bfloat16,
                load_in_8bit=model_args.quantization == "8bit",
                load_in_4bit=model_args.quantization == "4bit",
            )

            # Add LoRA adapters
            model = FastLanguageModel.get_peft_model(
                model,
                r=model_args.lora_r,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            return model, tokenizer

        except ImportError:
            logger.warning("Unsloth not available, falling back to standard training")
            use_unsloth = False

    # Standard HuggingFace training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    # Quantization config
    bnb_config = None
    if model_args.quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif model_args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA config
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(
    config_path: Optional[str] = None,
    model_args: Optional[ModelArguments] = None,
    data_args: Optional[DataArguments] = None,
    training_args: Optional[TrainingArguments] = None,
):
    """Run training.

    Args:
        config_path: Path to YAML config file
        model_args: Model arguments (overrides config)
        data_args: Data arguments (overrides config)
        training_args: Training arguments (overrides config)
    """
    # Load config if provided
    config = {}
    if config_path:
        config = load_config(config_path)

    # Set defaults from config
    if model_args is None:
        model_config = config.get("model", {})
        lora_config = config.get("lora", {})
        model_args = ModelArguments(
            model_name=model_config.get("name", "Qwen/Qwen3-VL-30B-A3B-Thinking"),
            quantization=model_config.get("quantization", "8bit"),
            lora_r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("alpha", 128),
            lora_dropout=lora_config.get("dropout", 0.05),
        )

    if data_args is None:
        train_config = config.get("training", {})
        data_args = DataArguments(
            train_file=train_config.get("train_file", "data/processed/train.jsonl"),
            eval_file=train_config.get("eval_file"),
            max_seq_length=train_config.get("max_seq_length", 8192),
        )

    if training_args is None:
        train_config = config.get("training", {})
        training_args = TrainingArguments(
            output_dir=config.get("output_dir", "checkpoints/chess-tutor-v1"),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 16),
            learning_rate=train_config.get("learning_rate", 1e-4),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            warmup_ratio=train_config.get("warmup_ratio", 0.03),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            logging_steps=train_config.get("logging_steps", 10),
            eval_strategy=train_config.get("eval_strategy", "steps"),
            eval_steps=train_config.get("eval_steps", 200),
            save_strategy=train_config.get("save_strategy", "steps"),
            save_steps=train_config.get("save_steps", 500),
            save_total_limit=train_config.get("save_total_limit", 3),
            bf16=train_config.get("bf16", True),
            gradient_checkpointing=train_config.get("gradient_checkpointing", True),
            seed=train_config.get("seed", 42),
            report_to="wandb" if config.get("wandb", {}).get("enabled") else "none",
        )

    # Initialize wandb
    if config.get("wandb", {}).get("enabled"):
        import wandb
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "chess-tutor"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
        )

    # Load data
    logger.info(f"Loading training data from {data_args.train_file}")
    train_dataset = load_training_data(data_args.train_file)
    train_dataset = train_dataset.map(format_messages_for_training)

    eval_dataset = None
    if data_args.eval_file:
        logger.info(f"Loading eval data from {data_args.eval_file}")
        eval_dataset = load_training_data(data_args.eval_file)
        eval_dataset = eval_dataset.map(format_messages_for_training)

    # Set up model
    logger.info(f"Loading model: {model_args.model_name}")
    model, tokenizer = setup_model_and_tokenizer(model_args)

    # Create trainer
    try:
        from unsloth import UnslothTrainer, UnslothTrainingArguments

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=UnslothTrainingArguments(
                **training_args.to_dict(),
            ),
            dataset_text_field="text",
            max_seq_length=data_args.max_seq_length,
        )
    except ImportError:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=data_args.max_seq_length,
        )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    logger.info("Training complete!")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Chess Tutor model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="training/configs/qwen3_vl_30b.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Training data file (overrides config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    args = parser.parse_args()

    # Load base config
    config = load_config(args.config) if Path(args.config).exists() else {}

    # Apply overrides
    if args.data:
        config.setdefault("training", {})["train_file"] = args.data
    if args.output:
        config["output_dir"] = args.output
    if args.epochs:
        config.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    # Save modified config temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name

    train(config_path=temp_config)

    # Cleanup
    os.unlink(temp_config)


if __name__ == "__main__":
    main()
