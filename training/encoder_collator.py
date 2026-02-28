import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from src.encoder.board_tensor import board_to_tensor
import chess

@dataclass
class EncoderDataCollator:
    """Collates token IDs and computes the spatial board representation on the fly.
    
    Extracts the FEN string from the training example metadata (if present)
    and passes it through the board_to_tensor helper. Returns the text tensors
    plus the stacked board_tensor batch for the ChessLMWithEncoder module.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pop the FEN out of features before we pass to base padding
        # FEN string should be stored in the example by format_dataset
        fens = []
        for feature in features:
            if "fen" not in feature:
                # Default to starting position if not provided
                fens.append(chess.STARTING_FEN)
            else:
                fens.append(feature.pop("fen"))
        
        # Base huggingface collation (pads input_ids, attention_mask, labels)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Build board tensors
        board_tensors = []
        for fen in fens:
            board = chess.Board(fen)
            b_tensor = board_to_tensor(board)
            board_tensors.append(b_tensor)
            
        # (BatchSize, 19, 8, 8)
        batch["board_tensor"] = torch.stack(board_tensors)
        
        return batch
