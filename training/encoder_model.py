"""Wrapper module coupling Qwen3-4B with the ChessEncoder trunk."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.encoder.cnn import ChessEncoder


class ChessLMWithEncoder(nn.Module):
    """Combines a base LLM (e.g., Qwen3-4B) with a ResNet board encoder.

    The encoder processes a 38-channel 8x8 spatial tensor (before+after board)
    and projects it into the LLM's embedding space. This 'soft token' is then
    prepended to the text embeddings before they are passed through
    the transformer layers.
    """

    def __init__(
        self,
        llm: nn.Module,
        hidden_size: int = 2560,
        cnn_hidden_size: int = 512,
        cnn_num_blocks: int = 15,
    ):
        super().__init__()
        self.llm = llm
        # 72M param ResNet (15 blocks, 512 filters, 38-ch input) -> hidden_size dim output
        self.cnn = ChessEncoder(
            in_channels=38,
            hidden_size=cnn_hidden_size,
            num_blocks=cnn_num_blocks,
            out_dim=hidden_size,
        )

        # Ensure the LLM embeddings are accessible
        if hasattr(self.llm, "get_input_embeddings"):
            self.embed_tokens = self.llm.get_input_embeddings()
        elif hasattr(self.llm.model, "embed_tokens"):
            self.embed_tokens = self.llm.model.embed_tokens
        else:
            raise ValueError("Could not find input embeddings layer in the LLM.")

    def forward(
        self,
        board_tensor: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass merging spatial and structural inputs.

        Args:
            board_tensor: (B, 19, 8, 8) float tensor of spatial board info.
            input_ids: (B, L) integer token IDs for the text prompt.
            attention_mask: (B, L) padding mask.
            labels: (B, L) target token IDs (or -100 to ignore).

        Returns:
            CausalLM output from the underlying model.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 1. Process board tensor -> single soft token per batch item
        # Outputs (B, hidden_size)
        board_emb = self.cnn(board_tensor)
        # Reshape to (B, 1, hidden_size) to prepend as a sequence token
        board_token = board_emb.unsqueeze(1).to(self.embed_tokens.weight.dtype)

        # 2. Get text embeddings -> (B, L, hidden_size)
        text_embs = self.embed_tokens(input_ids)

        # 3. Concatenate board token -> (B, 1 + L, hidden_size)
        inputs_embeds = torch.cat([board_token, text_embs], dim=1)

        # 4. Prepend attention mask with 1s (board token is always attended to)
        if attention_mask is not None:
            board_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([board_mask, attention_mask], dim=1)

        # 5. Prepend labels with -100 (never predict the board token)
        if labels is not None:
            board_labels = -100 * torch.ones((batch_size, 1), dtype=labels.dtype, device=device)
            labels = torch.cat([board_labels, labels], dim=1)

        # 6. Pass through the base LLM
        return self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def print_trainable_parameters(self):
        """Helper to verify parameter counts when wrapping with LoRA."""
        if hasattr(self.llm, "print_trainable_parameters"):
            self.llm.print_trainable_parameters()

        trainable_params, all_param = 0, 0
        for param in self.cnn.parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(
            f"Encoder params: trainable={trainable_params:,d} || "
            f"all={all_param:,d} || trainable%={100 * trainable_params / all_param:.4f}"
        )
