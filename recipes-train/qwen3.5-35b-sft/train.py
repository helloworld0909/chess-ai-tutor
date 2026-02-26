"""SFT training — Qwen3.5-35B-A3B + QLoRA (4-bit NF4, r=8).

Parallelism: pipeline parallel (device_map manual layer split across 2 GPUs).
  Single process — do NOT use torchrun for this recipe.

Usage (via recipe):
    ./recipes-train/qwen3.5-35b-sft/start.sh

Direct:
    python3 recipes-train/qwen3.5-35b-sft/train.py \\
        --config recipes-train/qwen3.5-35b-sft/config.yaml
"""

import argparse
import logging
import os
import sys

import torch

# Allow importing training.lib from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.lib import load_config, run_training  # noqa: E402

_logger = logging.getLogger(__name__)


def setup(config_path: str):
    """Load Qwen3.5-35B with 4-bit NF4 QLoRA, split evenly across 2 GPUs."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})

    model_name = model_cfg["model_name"]

    _logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Manual layer split across 2 GPUs — even split by layer count
    hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
    n_layers = text_cfg.num_hidden_layers
    split = n_layers // 2
    device_map = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = 0 if i < split else 1
    _logger.info("Layer split: GPU0=%d layers, GPU1=%d layers", split, n_layers - split)

    _logger.info("Loading model (4-bit NF4): %s", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="recipes-train/qwen3.5-35b-sft/config.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    args = parser.parse_args()

    model, tokenizer = setup(args.config)
    run_training(args.config, model, tokenizer, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
