"""Data collator for ChessLMWithEncoder — multi-move token splicing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chess
import torch
from transformers import PreTrainedTokenizerBase

from src.encoder import MOVE_TOKEN, MOVE_TOKEN_ID
from src.encoder.board_tensor import boards_to_tensor

_logger = logging.getLogger(__name__)


@dataclass
class EncoderDataCollator:
    """Collates token IDs and computes board tensors for every <|move|> token.

    Each training example contains one or more <|move|> sentinels injected by
    the dataset _fmt() function:
      - 1 token for the student's move (from metadata fen + move_san)
      - N tokens for the moves inside <line>...</line> blocks (replayed from
        the board state after the student's move)

    The collator produces:
      batch["board_tensors_flat"]: (N_total, 38, 8, 8) — all board tensors for
          the batch concatenated in the same order as the <|move|> tokens appear
          left-to-right in the padded input_ids.
      batch["move_counts"]: (B,) — number of move tensors per example, so the
          forward pass can validate alignment.

    line_sans_json format (produced by _fmt):
        JSON-serialised list[list[str]] — one inner list per <line> block,
        each inner list contains the SAN moves in order.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None

    def __post_init__(self) -> None:
        self._move_token_id: int = MOVE_TOKEN_ID

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pop non-tensor fields before HF padding (unknown keys cause errors)
        fens: List[str] = []
        move_sans: List[str] = []
        line_sans_lists: List[List[List[str]]] = []
        labels_list: List[List[int]] = []

        for feat in features:
            fens.append(feat.pop("fen", chess.STARTING_FEN))
            move_sans.append(feat.pop("move_san", ""))
            raw = feat.pop("line_sans_json", "[]")
            try:
                line_sans_lists.append(json.loads(raw))
            except Exception:
                line_sans_lists.append([])
            # Pop labels; tokenizer.pad() doesn't handle them correctly
            lbl = feat.pop("labels", None)
            if lbl is not None:
                labels_list.append(list(lbl))

        # Pad input_ids + attention_mask only
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad labels with -100 to match padded length
        if labels_list:
            max_len = batch["input_ids"].shape[1]
            padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels_list]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        all_tensors: List[torch.Tensor] = []
        move_counts: List[int] = []

        for b_idx, (fen, student_san, line_sans) in enumerate(
            zip(fens, move_sans, line_sans_lists)
        ):
            tensors_for_example: List[torch.Tensor] = []

            # --- 1. Student move tensor ---
            board = chess.Board(fen)
            student_move: Optional[chess.Move] = None
            if student_san:
                try:
                    student_move = board.parse_san(student_san)
                except Exception:
                    pass
            tensors_for_example.append(boards_to_tensor(board, student_move))

            # --- 2. LINE move tensors — replay from position after student move ---
            # Phase 2 compatibility: line replay starts from fen (pre-move board)
            for line_san_list in line_sans:
                line_board = board.copy()
                for san in line_san_list:
                    try:
                        mv = line_board.parse_san(san)
                        tensors_for_example.append(boards_to_tensor(line_board, mv))
                        line_board.push(mv)
                    except Exception:
                        # Illegal/unparseable SAN: use identity (null-move) tensor
                        tensors_for_example.append(boards_to_tensor(line_board, None))

            # --- 3. Validate count matches <|move|> tokens in input_ids ---
            n_tokens = (batch["input_ids"][b_idx] == self._move_token_id).sum().item()
            n_tensors = len(tensors_for_example)
            if n_tokens != n_tensors:
                _logger.debug(
                    "example %d: %d move tokens vs %d board tensors (truncation) — adjusting",
                    b_idx,
                    n_tokens,
                    n_tensors,
                )
                while len(tensors_for_example) < n_tokens:
                    tensors_for_example.append(boards_to_tensor(chess.Board(fen), None))
                tensors_for_example = tensors_for_example[:n_tokens]

            move_counts.append(len(tensors_for_example))
            all_tensors.extend(tensors_for_example)

        batch["board_tensors_flat"] = (
            torch.stack(all_tensors) if all_tensors else torch.zeros(0, 38, 8, 8)
        )
        batch["move_counts"] = torch.tensor(move_counts, dtype=torch.long)

        return batch
