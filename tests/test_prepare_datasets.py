"""Tests for data/pipeline/prepare_datasets.py.

All tests use mock data — no HuggingFace downloads or Stockfish required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "pipeline"))

from prepare_datasets import (
    AugmentedSample,
    ChessCotTransformer,
    IcannosTransformer,
    TextbookTransformer,
    dedup_samples,
    format_training_sample,
    split_and_write,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ── ChessCotTransformer ──────────────────────────────────────────────────────


def _make_cot_dataset(rows: list[dict]) -> MagicMock:
    """Create a mock HF dataset from rows."""
    ds = MagicMock()
    ds.__iter__ = lambda self: iter(rows)
    return ds


def test_chess_cot_extracts_valid_sample():
    rows = [
        {
            "fen": STARTING_FEN,
            "board": "",
            "turn": "w",
            "reasoning": "This is a good opening. The pawn controls the center.",
            "response": "e2e4",
            "reward": 0.9,
        }
    ]
    with patch("datasets.load_dataset", return_value=_make_cot_dataset(rows)):
        tx = ChessCotTransformer()
        samples = list(tx.extract(max_samples=10))

    assert len(samples) == 1
    assert samples[0].move_uci == "e2e4"
    assert samples[0].move_san == "e4"
    assert samples[0].source == "chess_cot"


def test_chess_cot_filters_low_reward():
    rows = [
        {
            "fen": STARTING_FEN,
            "reasoning": "Random.",
            "response": "e2e4",
            "reward": 0.1,  # below threshold
        }
    ]
    with patch("datasets.load_dataset", return_value=_make_cot_dataset(rows)):
        tx = ChessCotTransformer()
        samples = list(tx.extract())

    assert len(samples) == 0


def test_chess_cot_skips_illegal_move():
    rows = [
        {
            "fen": STARTING_FEN,
            "reasoning": "Let me try.",
            "response": "e1e5",  # illegal
            "reward": 0.9,
        }
    ]
    with patch("datasets.load_dataset", return_value=_make_cot_dataset(rows)):
        tx = ChessCotTransformer()
        samples = list(tx.extract())

    assert len(samples) == 0


def test_chess_cot_respects_max_samples():
    rows = [
        {"fen": STARTING_FEN, "reasoning": "Move one.", "response": "e2e4", "reward": 0.9},
        {"fen": STARTING_FEN, "reasoning": "Move two.", "response": "d2d4", "reward": 0.9},
        {"fen": STARTING_FEN, "reasoning": "Move three.", "response": "g1f3", "reward": 0.9},
    ]
    with patch("datasets.load_dataset", return_value=_make_cot_dataset(rows)):
        tx = ChessCotTransformer()
        samples = list(tx.extract(max_samples=2))

    assert len(samples) == 2


# ── IcannosTransformer ────────────────────────────────────────────────────────

SAMPLE_PGN = (
    '[Event "Test Study"]\n'
    '[Site "https://lichess.org/study/test"]\n'
    "\n"
    "1. e4 { This is an excellent opening move because it controls the center. } "
    "e5 2. Nf3 Nc6 *\n"
)


def _make_csv_response(pgn_texts: list[str]) -> MagicMock:
    """Create a mock httpx response with CSV content."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["split", "text"])
    writer.writeheader()
    for t in pgn_texts:
        writer.writerow({"split": "train", "text": t})
    mock_resp = MagicMock()
    mock_resp.text = output.getvalue()
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_icannos_extracts_annotated_moves():
    # Mock returns same CSV for both URLs → 2 samples (one per CSV file)
    with patch("httpx.get", return_value=_make_csv_response([SAMPLE_PGN])):
        tx = IcannosTransformer()
        samples = list(tx.extract())

    assert len(samples) >= 1
    assert samples[0].move_san == "e4"
    assert "center" in samples[0].coaching_text
    assert samples[0].source == "icannos_studies"


def test_icannos_skips_short_annotations():
    pgn = '[Event "Test"]\n\n1. e4 { ok } e5 *\n'
    with patch("httpx.get", return_value=_make_csv_response([pgn])):
        tx = IcannosTransformer()
        samples = list(tx.extract())

    assert len(samples) == 0  # "ok" is too short and has no keywords


# ── TextbookTransformer ───────────────────────────────────────────────────────


def test_textbook_reads_augmented_jsonl(tmp_path: Path):
    data = {
        "fen": STARTING_FEN,
        "move_uci": "e2e4",
        "move_san": "e4",
        "annotation": "A strong central pawn push that controls key squares.",
        "concepts": ["center control"],
    }
    augmented = tmp_path / "augmented.jsonl"
    augmented.write_text(json.dumps(data) + "\n")

    tx = TextbookTransformer(augmented)
    samples = list(tx.extract())

    assert len(samples) == 1
    assert samples[0].move_san == "e4"
    assert "center control" in samples[0].thinking_text


def test_textbook_handles_missing_file(tmp_path: Path):
    tx = TextbookTransformer(tmp_path / "nonexistent.jsonl")
    samples = list(tx.extract())
    assert len(samples) == 0


# ── format_training_sample ────────────────────────────────────────────────────


def _make_augmented_sample(**overrides) -> AugmentedSample:
    defaults = dict(
        fen=STARTING_FEN,
        move_uci="e2e4",
        move_san="e4",
        coaching_text="Good opening move controlling the center.",
        thinking_text="The center pawn advances.",
        source="test",
        classification="Best",
        eval_str="+0.35",
        best_move_san="e4",
        cp_loss=0,
        candidates=["e4 (+0.35)", "d4 (+0.30)"],
        opponent_threats=[],
    )
    defaults.update(overrides)
    return AugmentedSample(**defaults)


def test_format_training_sample_has_three_messages():
    sample = format_training_sample(_make_augmented_sample())
    assert len(sample["messages"]) == 3
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_format_training_sample_user_has_board():
    sample = format_training_sample(_make_augmented_sample())
    user = sample["messages"][1]["content"]
    assert "a b c d e f g h" in user


def test_format_training_sample_user_has_classification():
    sample = format_training_sample(_make_augmented_sample(classification="Blunder"))
    user = sample["messages"][1]["content"]
    assert "Classification: Blunder" in user


def test_format_training_sample_assistant_has_think_block():
    sample = format_training_sample(_make_augmented_sample(thinking_text="Let me analyze this."))
    assistant = sample["messages"][2]["content"]
    assert "<think>" in assistant
    assert "Let me analyze this." in assistant


def test_format_training_sample_no_think_when_empty():
    sample = format_training_sample(_make_augmented_sample(thinking_text=""))
    assistant = sample["messages"][2]["content"]
    assert "<think>" not in assistant


def test_format_training_sample_has_metadata():
    sample = format_training_sample(_make_augmented_sample(source="chess_cot"))
    assert sample["metadata"]["source"] == "chess_cot"
    assert sample["metadata"]["classification"] == "Best"


# ── dedup_samples ─────────────────────────────────────────────────────────────


def test_dedup_removes_duplicates():
    s1 = _make_augmented_sample(coaching_text="short")
    s2 = _make_augmented_sample(coaching_text="a much longer coaching text here")
    deduped = dedup_samples([s1, s2])
    assert len(deduped) == 1
    assert deduped[0].coaching_text == "a much longer coaching text here"


def test_dedup_keeps_different_moves():
    s1 = _make_augmented_sample(move_uci="e2e4")
    s2 = _make_augmented_sample(move_uci="d2d4")
    deduped = dedup_samples([s1, s2])
    assert len(deduped) == 2


# ── split_and_write ───────────────────────────────────────────────────────────


def test_split_and_write_creates_files(tmp_path: Path):
    samples = [_make_augmented_sample() for _ in range(20)]
    train_n, eval_n = split_and_write(samples, tmp_path, eval_fraction=0.2)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()
    assert train_n + eval_n == 20
    assert eval_n >= 1

    # Verify JSONL format
    with open(tmp_path / "train.jsonl") as f:
        for line in f:
            sample = json.loads(line)
            assert "messages" in sample
            assert len(sample["messages"]) == 3


def test_split_and_write_deterministic(tmp_path: Path):
    samples = [_make_augmented_sample(move_uci=f"e2e{i}") for i in range(3, 7)]  # make unique
    # Won't have valid chess moves but format_training_sample just needs FEN + UCI
    # Actually these are invalid moves — let me use valid ones
    samples = [_make_augmented_sample(coaching_text=f"Version {i}") for i in range(10)]

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    split_and_write(samples[:], dir1, seed=42)
    split_and_write(samples[:], dir2, seed=42)

    t1 = (dir1 / "train.jsonl").read_text()
    t2 = (dir2 / "train.jsonl").read_text()
    assert t1 == t2
