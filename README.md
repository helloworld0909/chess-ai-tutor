# Chess AI Tutor

A pedagogical chess analysis system combining Stockfish 17 NNUE evaluations with fine-tuned LLM explanations via MCP integration.

## Features

- **Move Analysis**: Classify moves as Best/Great/Good/Inaccuracy/Mistake/Blunder
- **Natural Language Explanations**: Human-like explanations of chess concepts
- **MCP Integration**: Stockfish tools accessible via Model Context Protocol
- **Vision-Language Support**: Process board images with Qwen3-VL model
- **Verification Loop**: Cross-check LLM outputs against engine analysis

## Installation

```bash
# Install with uv
uv sync

# Install Stockfish 17
./scripts/install_stockfish.sh

# Run tests
uv run pytest tests/ -v
```

## Quick Start

```bash
# Set Stockfish path
export PATH="$HOME/.local/bin:$PATH"

# Run CLI tutor
uv run python -c "import sys; sys.path.insert(0,'src'); from tutor.cli import main; main()"
```

---

## Project Structure & Implementation Status

### `src/chess_mcp/` - MCP Server & Stockfish Integration ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `__init__.py` | Package exports | ✅ Done |
| `stockfish.py` | Async Stockfish UCI wrapper | ✅ Done |
| `tools.py` | MCP tool implementations | ✅ Done |
| `server.py` | MCP server entry point | ✅ Done |
| `representations.py` | FEN/ASCII/SVG/PNG converters | ✅ Done |

**Implemented Features:**
- Async Stockfish process management with UCI protocol
- 6 MCP tools: `get_best_move`, `get_eval`, `get_threats`, `compare_moves`, `get_legal_moves`, `validate_move`
- Multi-PV analysis support
- Score types: centipawns, mate distance, win probability
- Position representations: FEN, ASCII board, piece-square list, SVG/PNG rendering

**Usage:**
```python
from chess_mcp.stockfish import Stockfish
async with Stockfish() as sf:
    analysis = await sf.analyze("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", depth=20)
    print(analysis.best_move, analysis.score)
```

---

### `src/verification/` - Move Validation & LLM Verification ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `__init__.py` | Package exports | ✅ Done |
| `legality.py` | Move legality validation with python-chess | ✅ Done |
| `tactical_loop.py` | LLM output verification against engine | ✅ Done |

**Implemented Features:**
- Move validation in UCI, SAN, and LAN formats
- Flexible move parsing (auto-detect format)
- Game state detection (check, checkmate, stalemate)
- Move classification by centipawn loss thresholds
- LLM response verification against Stockfish
- Classification extraction from natural language text

**Usage:**
```python
from verification.legality import validate_move, MoveFormat
result = validate_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4", MoveFormat.SAN)
print(result.valid, result.move_uci)  # True, e2e4
```

---

### `src/tutor/` - CLI Interface ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `__init__.py` | Package exports | ✅ Done |
| `cli.py` | Interactive CLI tutor | ✅ Done |

**Implemented Features:**
- Interactive chess session with Rich terminal UI
- Board display with ASCII art
- Engine analysis on demand
- Move comparison and classification
- Human-like explanations for moves
- Game history and undo support

**Commands:**
- `<move>` - Make a move (e.g., `e4`, `Nf3`, `O-O`)
- `analyze [move]` - Analyze position or specific move
- `undo` - Undo last move
- `moves` - Show legal moves
- `info` - Show position details
- `new [fen]` - Start new game
- `quit` - Exit

---

### `data/pipeline/` - Training Data Pipeline ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `collect_textbooks.py` | Download PGNs from various sources | ✅ Done |
| `extract_annotations.py` | Parse PGN annotations | ✅ Done |
| `augment_stockfish.py` | Add engine evaluations | ✅ Done |
| `render_boards.py` | Generate board images for VL training | ✅ Done |

**Data Sources Configured:**
- PGN Mentor (annotated GM games)
- Lichess Open Database
- Lichess Elite DB (2400+ rated)
- Lichess Studies API
- Chessgames.com collections

**Pipeline Steps:**
```bash
# 1. Collect PGNs
python data/pipeline/collect_textbooks.py -o data/raw

# 2. Extract annotations
python data/pipeline/extract_annotations.py -i data/raw -o data/processed/annotations.jsonl

# 3. Augment with Stockfish
python data/pipeline/augment_stockfish.py --create-training

# 4. Render board images
python data/pipeline/render_boards.py --size 448
```

---

### `training/` - Fine-Tuning Setup ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `configs/qwen3_vl_30b.yaml` | Training configuration | ✅ Done |
| `train.py` | Unsloth/TRL training script | ✅ Done |

**Model Configuration:**
- Base: `Qwen/Qwen3-VL-30B-A3B-Thinking`
- Quantization: 8-bit (bitsandbytes)
- Method: LoRA (r=64, alpha=128)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Training:**
```bash
python training/train.py --config training/configs/qwen3_vl_30b.yaml
```

---

### `scripts/` - Utility Scripts ✅ COMPLETE

| File | Description | Status |
|------|-------------|--------|
| `install_stockfish.sh` | Download & install Stockfish 17 | ✅ Done |

---

### `tests/` - Unit Tests ✅ COMPLETE (98 tests)

| File | Tests | Description |
|------|-------|-------------|
| `test_stockfish.py` | 16 | Stockfish wrapper tests |
| `test_mcp_tools.py` | 17 | MCP tool tests |
| `test_verification.py` | 40 | Legality & classification tests |
| `test_representations.py` | 25 | Position representation tests |

**Run tests:**
```bash
uv run pytest tests/ -v
```

---

## TODO / Not Yet Implemented

### `src/tutor/` - Additional Files Needed
| File | Description | Status |
|------|-------------|--------|
| `classifier.py` | Move classification with fine-tuned model | ❌ TODO |
| `explainer.py` | Natural language explanation generator | ❌ TODO |
| `session.py` | Game state & conversation history | ❌ TODO |

### `training/` - Additional Files Needed
| File | Description | Status |
|------|-------------|--------|
| `eval_chess.py` | Elo evaluation harness | ❌ TODO |
| `merge_adapter.py` | Merge LoRA adapter with base model | ❌ TODO |
| `configs/lora_config.yaml` | Separate LoRA config | ❌ TODO |
| `configs/ds_zero3.json` | DeepSpeed ZeRO-3 config | ❌ TODO |

### `data/pipeline/` - Additional Files Needed
| File | Description | Status |
|------|-------------|--------|
| `validate_dataset.py` | Dataset validation & statistics | ❌ TODO |

### Integration Tasks
- [ ] Connect fine-tuned model to CLI tutor
- [ ] Implement conversation memory for follow-up questions
- [ ] Add web UI (optional, after CLI is complete)
- [ ] Create evaluation benchmark

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Tutor                                │
│                      (src/tutor/cli.py)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Verification   │    │  Fine-tuned LLM │                    │
│  │  (legality.py)  │    │  (Qwen3-VL-30B) │                    │
│  │  (tactical.py)  │    │                 │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────┐                   │
│  │           MCP Server (6 tools)           │                   │
│  │         (src/chess_mcp/server.py)        │                   │
│  └────────────────────┬────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────┐                   │
│  │         Stockfish 17 (NNUE)             │                   │
│  │       (src/chess_mcp/stockfish.py)       │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

Environment variables (see `.env.example`):
```bash
STOCKFISH_PATH=/home/zheng/.local/bin/stockfish
STOCKFISH_DEPTH=20
STOCKFISH_THREADS=4
STOCKFISH_HASH_MB=256
```
