import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from encoder.cnn import ChessEncoder, ResidualBlock
from encoder.board_tensor import board_to_tensor
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.encoder_model import ChessLMWithEncoder

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
    encoder = ChessEncoder(in_channels=19, hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 19, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 256)

def test_board_to_tensor():
    board = chess.Board() # initial position
    tensor = board_to_tensor(board)
    assert tensor.shape == (19, 8, 8)
    
    # Check white pieces channel mapping
    # Rank 0 (a1-h1) should have rooks (channel 3)
    assert tensor[3, 0, 0] == 1.0 # Ra1
    assert tensor[3, 0, 7] == 1.0 # Rh1
    
    # Check side to move (White)
    assert torch.all(tensor[12] == 1.0)
    
    # Black pieces
    assert tensor[6 + 3, 7, 0] == 1.0 # Ra8
    
    # Move num = 1 (1 / 200 = 0.005)
    assert torch.allclose(tensor[18], torch.tensor(0.005))

def test_chess_lm_with_encoder(mock_llm, mock_tokenizer):
    wrapper = ChessLMWithEncoder(llm=mock_llm, hidden_size=256)
    
    # 2 batch items, seq length 5
    batch_size = 2
    seq_len = 5
    board_tensor = torch.randn(batch_size, 19, 8, 8)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    out = wrapper(
        board_tensor=board_tensor,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    
    # Return from Qwen is an output object with `.logits`
    # shape: (Batch, SeqLen + 1_board_token, VocabSize)
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (batch_size, seq_len + 1)
