import json
import os
import sys
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import chess

from encoder.board_tensor import board_to_tensor, boards_to_tensor
from encoder.cnn import ChessEncoder, ResidualBlock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.encoder_model import ChessLMWithEncoder
from training.train_encoder_pretrain import EncoderPretrainDataset, EncoderRegressorHead


@pytest.fixture
def mock_tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def mock_llm():
    # Only load config + untrained mini embeddings to avoid huge local loads for a fast test
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    config.hidden_size = 256
    config.num_hidden_layers = 1
    llm = AutoModelForCausalLM.from_config(config)
    return llm


def test_residual_block():
    block = ResidualBlock(channels=64)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_chess_encoder_shape():
    # Default: 38-channel (before + after)
    encoder = ChessEncoder(hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 38, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_chess_encoder_legacy_19ch():
    # Legacy single-board mode still works
    encoder = ChessEncoder(in_channels=19, hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 19, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_board_to_tensor():
    board = chess.Board()  # initial position
    tensor = board_to_tensor(board)
    assert tensor.shape == (19, 8, 8)

    # Check white pieces channel mapping
    # Rank 0 (a1-h1) should have rooks (channel 3)
    assert tensor[3, 0, 0] == 1.0  # Ra1
    assert tensor[3, 0, 7] == 1.0  # Rh1

    # Check side to move (White)
    assert torch.all(tensor[12] == 1.0)

    # Black pieces
    assert tensor[6 + 3, 7, 0] == 1.0  # Ra8

    # Move num = 1 (1 / 200 = 0.005)
    assert torch.allclose(tensor[18], torch.tensor(0.005))


def test_boards_to_tensor_with_move():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    tensor = boards_to_tensor(board, move)
    assert tensor.shape == (38, 8, 8)
    # Before channels: e2 pawn present (channel 0 = white pawn, rank=1, file=4)
    assert tensor[0, 1, 4] == 1.0
    # After channels: e2 pawn gone (channel 19+0, rank=1, file=4)
    assert tensor[19, 1, 4] == 0.0
    # After channels: e4 pawn present (channel 19+0, rank=3, file=4)
    assert tensor[19, 3, 4] == 1.0


def test_boards_to_tensor_static():
    # No move — identity convention: channels 19-37 == channels 0-18
    board = chess.Board()
    tensor = boards_to_tensor(board, move=None)
    assert tensor.shape == (38, 8, 8)
    assert torch.equal(tensor[:19], tensor[19:])


def test_chess_lm_with_encoder(mock_llm, mock_tokenizer):
    wrapper = ChessLMWithEncoder(llm=mock_llm, hidden_size=256)

    # 2 batch items, seq length 5 — now using 38-channel board tensors
    batch_size = 2
    seq_len = 5
    board_tensor = torch.randn(batch_size, 38, 8, 8)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    out = wrapper(board_tensor=board_tensor, input_ids=input_ids, attention_mask=attention_mask)

    # Return from Qwen is an output object with `.logits`
    # shape: (Batch, SeqLen + 1_board_token, VocabSize)
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (batch_size, seq_len + 1)


# ---------------------------------------------------------------------------
# EncoderRegressorHead
# ---------------------------------------------------------------------------


def test_encoder_regressor_head_shape():
    encoder = ChessEncoder(in_channels=38, hidden_size=32, num_blocks=2, out_dim=64)
    head = EncoderRegressorHead(encoder)
    x = torch.randn(4, 38, 8, 8)
    preds = head(x)
    assert preds.shape == (4,)
    # tanh output must be in (-1, 1)
    assert preds.abs().max().item() < 1.0


def test_encoder_regressor_head_range():
    """Predictions must stay in (-1, 1) regardless of input magnitude."""
    encoder = ChessEncoder(in_channels=38, hidden_size=32, num_blocks=2, out_dim=64)
    head = EncoderRegressorHead(encoder)
    # Large random input — tanh should still clamp
    x = torch.randn(8, 38, 8, 8) * 100.0
    preds = head(x)
    assert (preds > -1.0).all()
    assert (preds < 1.0).all()


# ---------------------------------------------------------------------------
# EncoderPretrainDataset
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "move_uci": "e7e5",
        "cp_diff_scaled": -0.05,
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "move_uci": "g1f3",
        "cp_diff_scaled": 0.12,
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "move_uci": "b8c6",
        "cp_diff_scaled": 0.03,
    },
]


def test_encoder_pretrain_dataset_len():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path)
    assert len(ds) == len(_SAMPLE_RECORDS)
    os.unlink(tmp_path)


def test_encoder_pretrain_dataset_item_shapes():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path)
    board_tensor, label = ds[0]
    assert board_tensor.shape == (38, 8, 8)
    assert label.shape == ()
    assert label.dtype == torch.float32
    os.unlink(tmp_path)


def test_encoder_pretrain_dataset_limit():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path, limit=2)
    assert len(ds) == 2
    os.unlink(tmp_path)
