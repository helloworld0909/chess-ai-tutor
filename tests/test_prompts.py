"""Tests for tutor.prompts — shared prompt utilities."""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tutor.prompts import (
    SYSTEM_PROMPT,
    board_ascii,
    format_user_prompt,
)

# ── board_ascii ───────────────────────────────────────────────────────────────


def test_board_ascii_has_file_labels():
    result = board_ascii(chess.Board())
    assert "a b c d e f g h" in result


def test_board_ascii_has_rank_labels():
    result = board_ascii(chess.Board())
    for rank in "12345678":
        assert rank in result


def test_board_ascii_white_to_move():
    result = board_ascii(chess.Board())
    assert "White to move" in result


def test_board_ascii_black_to_move():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    result = board_ascii(board)
    assert "Black to move" in result


# ── format_user_prompt ────────────────────────────────────────────────────────


def test_format_user_prompt_includes_board():
    board = chess.Board()
    prompt = format_user_prompt(
        board_ascii_str=board_ascii(board),
        san="e4",
        classification="Best",
        eval_str="+0.35",
    )
    assert "a b c d e f g h" in prompt
    assert "Move: e4" in prompt
    assert "Classification: Best" in prompt


def test_format_user_prompt_includes_candidates():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="Nf3",
        classification="Best",
        eval_str="+0.20",
        candidates=["Nf3 (+0.20)", "d4 (+0.18)", "e4 (+0.15)"],
    )
    assert "Engine top candidates: Nf3" in prompt


def test_format_user_prompt_includes_threats():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="h3",
        classification="Good",
        eval_str="+0.10",
        opponent_threats=["Bg4"],
    )
    assert "Opponent threats if you passed: Bg4" in prompt


def test_format_user_prompt_best_line_only_when_cp_loss():
    prompt_best = format_user_prompt(
        board_ascii_str="",
        san="e4",
        classification="Best",
        eval_str="+0.35",
        cp_loss=0,
    )
    assert "Engine best was" not in prompt_best

    prompt_bad = format_user_prompt(
        board_ascii_str="",
        san="f3",
        classification="Mistake",
        eval_str="+0.35",
        best_move="e4",
        cp_loss=200,
    )
    assert "Engine best was: e4" in prompt_bad


def test_format_user_prompt_ends_with_instruction():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="e4",
        classification="Best",
        eval_str="+0.35",
    )
    assert "chess idea" in prompt or "Explain" in prompt


def test_format_user_prompt_has_before_after_sections():
    prompt = format_user_prompt(
        board_ascii_str=board_ascii(chess.Board()),
        san="e4",
        classification="Best",
        eval_str="+0.35",
        fen=chess.STARTING_FEN,
    )
    assert "## Position before your move" in prompt
    assert "## Position after your move" in prompt


def test_format_user_prompt_no_verified_facts_section():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="e4",
        classification="Best",
        eval_str="+0.35",
    )
    assert "Verified Move Facts" not in prompt


# ── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────


def test_system_prompt_is_nonempty():
    assert len(SYSTEM_PROMPT) > 50


def test_system_prompt_mentions_classification():
    assert "classification" in SYSTEM_PROMPT.lower()
