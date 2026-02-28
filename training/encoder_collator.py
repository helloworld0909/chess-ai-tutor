from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chess
import torch
from transformers import PreTrainedTokenizerBase

from src.encoder.board_tensor import boards_to_tensor


@dataclass
class EncoderDataCollator:
    """Collates token IDs and computes the spatial board representation on the fly.

    Extracts the FEN and move_san from the training example metadata (if present)
    and passes them through boards_to_tensor to produce a (38, 8, 8) tensor:
      channels  0-18: board BEFORE the move (19-channel AlphaZero encoding)
      channels 19-37: board AFTER the move  (19-channel AlphaZero encoding)

    When no move is available (static position query), channels 19-37 == channels 0-18
    (identity / null-move convention — the delta is zero so the CNN learns bare position).

    The resulting board_tensor is prepended to the LLM input sequence as a single
    soft token by ChessLMWithEncoder before the transformer layers.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pop FEN and move_san before HF padding (unknown keys cause errors)
        fens: List[str] = []
        move_sans: List[Optional[str]] = []
        for feature in features:
            fens.append(feature.pop("fen", chess.STARTING_FEN))
            move_sans.append(feature.pop("move_san", None))

        # Base HuggingFace collation (pads input_ids, attention_mask, labels)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build (38, 8, 8) board tensors — before + after
        board_tensors = []
        for fen, move_san in zip(fens, move_sans):
            board = chess.Board(fen)
            move: Optional[chess.Move] = None
            if move_san:
                try:
                    move = board.parse_san(move_san)
                except Exception:
                    pass  # fall back to identity (no move)
            board_tensors.append(boards_to_tensor(board, move))

        # (BatchSize, 38, 8, 8)
        batch["board_tensor"] = torch.stack(board_tensors)

        return batch
