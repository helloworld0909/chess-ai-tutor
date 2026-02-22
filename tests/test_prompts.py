"""Tests for tutor.prompts — shared prompt utilities."""

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tutor.prompts import (
    SYSTEM_PROMPT,
    board_ascii,
    format_user_prompt,
    move_facts,
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


# ── move_facts ────────────────────────────────────────────────────────────────


def test_move_facts_capture():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    facts = move_facts(board, chess.Move.from_uci("e4d5"))
    assert any("captures pawn" in f for f in facts)


def test_move_facts_check():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("a1a8"))
    assert any("check" in f for f in facts)


def test_move_facts_castling():
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    facts = move_facts(board, chess.Move.from_uci("e1g1"))
    assert any("castles" in f for f in facts)


def test_move_facts_returns_list_for_quiet_move():
    facts = move_facts(chess.Board(), chess.Move.from_uci("e2e4"))
    assert isinstance(facts, list)


def test_move_facts_empty_for_missing_piece():
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("e2e4"))
    assert facts == []


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
    assert "Move played: e4" in prompt
    assert "Classification: Best" in prompt


def test_format_user_prompt_includes_candidates():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="Nf3",
        classification="Best",
        eval_str="+0.20",
        candidates=["Nf3 (+0.20)", "d4 (+0.18)", "e4 (+0.15)"],
    )
    assert "Engine's top candidates: Nf3" in prompt


def test_format_user_prompt_includes_threats():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="h3",
        classification="Good",
        eval_str="+0.10",
        opponent_threats=["Bg4"],
    )
    assert "Opponent's threats if you passed: Bg4" in prompt


def test_format_user_prompt_includes_facts():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="exd5",
        classification="Best",
        eval_str="+0.50",
        facts=["captures pawn on d5"],
    )
    assert "Verified move facts:" in prompt
    assert "- captures pawn on d5" in prompt


def test_format_user_prompt_best_line_only_when_cp_loss():
    prompt_best = format_user_prompt(
        board_ascii_str="",
        san="e4",
        classification="Best",
        eval_str="+0.35",
        cp_loss=0,
    )
    assert "Engine's best move was" not in prompt_best

    prompt_bad = format_user_prompt(
        board_ascii_str="",
        san="f3",
        classification="Mistake",
        eval_str="+0.35",
        best_move="e4",
        cp_loss=200,
    )
    assert "Engine's best move was: e4" in prompt_bad


def test_format_user_prompt_ends_with_instruction():
    prompt = format_user_prompt(
        board_ascii_str="",
        san="e4",
        classification="Best",
        eval_str="+0.35",
    )
    assert "Write 2-3 sentences" in prompt


# ── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────


def test_system_prompt_is_nonempty():
    assert len(SYSTEM_PROMPT) > 50


def test_system_prompt_mentions_classification():
    assert "classification" in SYSTEM_PROMPT.lower()
