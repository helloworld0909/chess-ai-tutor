"""Encoder training — Qwen3-4B + ChessEncoder (12.5M params) + QLoRA.

Integrates the (19, 8, 8) Board Tensor as a prepended soft token.
"""

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder
from training.lib import (
    load_config,
    load_jsonl_lines,
    make_training_args,
)

_logger = logging.getLogger(__name__)


def setup_encoder_model(config_path: str):
    """Load Qwen3-4B with 8-bit QLoRA and wrap it with ChessEncoder."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    encoder_cfg = config.get("encoder", {})
    model_name = model_cfg["model_name"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    _logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _logger.info("Loading base LLM (8-bit) on GPU %d", local_rank)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": local_rank},
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    peft_llm = get_peft_model(base_llm, lora_config)

    _logger.info("Wrapping with CNN Encoder...")
    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_hidden_size=encoder_cfg.get("hidden_size", 512),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 15),
    )
    model.to(torch.bfloat16)
    model.to(f"cuda:{local_rank}")

    # Load pretrained CNN weights if provided
    encoder_weights_path = encoder_cfg.get("pretrained_weights")
    if encoder_weights_path:
        _logger.info("Loading pretrained encoder weights from %s", encoder_weights_path)
        state = torch.load(
            encoder_weights_path, map_location=f"cuda:{local_rank}", weights_only=True
        )
        missing, unexpected = model.cnn.load_state_dict(state, strict=True)
        if missing:
            _logger.warning("Missing keys in encoder weights: %s", missing)
        if unexpected:
            _logger.warning("Unexpected keys in encoder weights: %s", unexpected)
        _logger.info("Pretrained encoder weights loaded successfully.")
    else:
        _logger.warning(
            "No encoder.pretrained_weights set in config — CNN starts from random init!"
        )

    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="training/configs/qwen3_4b_encoder_sft.yaml")
    parser.add_argument("--resume", nargs="?", const=True, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    wandb_cfg = config.get("wandb", {})

    if wandb_cfg.get("enabled"):
        import wandb

        wandb.init(
            project=wandb_cfg.get("project", "chess-tutor"),
            name=wandb_cfg.get("name", "encoder-sft-4b"),
            tags=wandb_cfg.get("tags", ["encoder-sft"]),
        )

    model, tokenizer = setup_encoder_model(args.config)

    keep_think = train_cfg.get("keep_think", True)
    _logger.info("Loading training data from %s", train_cfg["train_file"])

    def _fmt(example):
        messages = example["messages"] if keep_think else []
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        meta = example.get("metadata", {})
        return {
            "text": text,
            "fen": meta.get("fen", ""),
            "move_san": meta.get("move_san", ""),
        }

    raw_train = load_jsonl_lines(train_cfg["train_file"])
    from datasets import Dataset

    train_dataset = (
        Dataset.from_list(raw_train)
        .map(_fmt, num_proc=4)
        .select_columns(["text", "fen", "move_san"])
    )

    eval_dataset = None
    if train_cfg.get("eval_file"):
        raw_eval = load_jsonl_lines(train_cfg["eval_file"])
        eval_dataset = (
            Dataset.from_list(raw_eval)
            .map(_fmt, num_proc=4)
            .select_columns(["text", "fen", "move_san"])
        )

    training_args = make_training_args(config)

    # Remove packing because it breaks 1-to-1 fens <-> tokens mapping requirement
    if train_cfg.get("packing", False):
        _logger.warning(
            "Packing disabled because Encoder requires 1 position = 1 sequence mapping."
        )

    from trl import SFTTrainer
    from trl.trainer.sft_config import SFTConfig

    data_collator = EncoderDataCollator(tokenizer=tokenizer)

    dict_args = training_args.to_dict()
    dict_args["hub_token"] = training_args.hub_token
    dict_args.pop("push_to_hub_token", None)
    sft_config = SFTConfig(
        **dict_args,
        max_length=train_cfg.get("max_seq_length", 1600),
        dataset_text_field="text",
        packing=False,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    _logger.info("Starting encoder integration training...")
    trainer.train(resume_from_checkpoint=args.resume)

    _logger.info("Saving model to %s", training_args.output_dir)
    # This will save the PEFT layout + CNN weights via state_dict magic natively
    # handled by PyTorch since it's a unified nn.Module
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    _logger.info("Done.")


if __name__ == "__main__":
    main()
