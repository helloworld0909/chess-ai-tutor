"""Tests for tutor.analysis — compute_position_context and related helpers."""

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tutor.analysis import compute_position_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _board(fen: str) -> chess.Board:
    return chess.Board(fen)


# Starting position
START_FEN = chess.STARTING_FEN

# Endgame: K+P vs K
KP_FEN = "8/8/8/8/8/8/4P3/4K2k w - - 0 1"

# Position with passed pawn on e5 for white, open d-file, half-open e-file for black
PASSED_FEN = "r1bqkb1r/pp3ppp/2np1n2/4p3/2B1P3/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 7"

# Position where white has doubled f-pawns
DOUBLED_FEN = "rnbqkb1r/pppp1ppp/5n2/4p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"


# ---------------------------------------------------------------------------
# Game phase detection
# ---------------------------------------------------------------------------


def test_start_position_is_opening():
    ctx = compute_position_context(chess.Board())
    assert ctx["game_phase"] == "Opening"


def test_endgame_phase():
    ctx = compute_position_context(_board(KP_FEN))
    assert ctx["game_phase"] == "Endgame"


def test_move_number_start():
    ctx = compute_position_context(chess.Board())
    assert ctx["move_number"] == 1


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


def test_starting_material_equal():
    ctx = compute_position_context(chess.Board())
    assert ctx["material_white"] == ctx["material_black"]
    assert ctx["material_balance"] == 0


def test_starting_material_value():
    # Q=9 + 2R=10 + 2B=6 + 2N=6 + 8P=8 = 39
    ctx = compute_position_context(chess.Board())
    assert ctx["material_white"] == 39


def test_material_balance_after_capture():
    # White has captured black's queen: Qd8 is gone
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # remove black queen
    ctx = compute_position_context(board)
    assert ctx["material_balance"] == 9


# ---------------------------------------------------------------------------
# Pawn structure: passed pawns
# ---------------------------------------------------------------------------


def test_no_passed_pawns_at_start():
    ctx = compute_position_context(chess.Board())
    assert ctx["passed_pawns_white"] == []
    assert ctx["passed_pawns_black"] == []


def test_passed_pawn_detected():
    # Lone white pawn on e5 with no opposing pawn blocking/adjacent
    board = chess.Board("8/8/8/4P3/8/8/8/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "e5" in ctx["passed_pawns_white"]


def test_blocked_pawn_not_passed():
    # White pawn on e5, black pawn on e6 — not passed
    board = chess.Board("8/8/4p3/4P3/8/8/8/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "e5" not in ctx["passed_pawns_white"]


def test_adjacent_pawn_blocks_passed():
    # White pawn on e5, black pawn on f6 (adjacent file, in front) — not passed
    board = chess.Board("8/8/5p2/4P3/8/8/8/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "e5" not in ctx["passed_pawns_white"]


# ---------------------------------------------------------------------------
# Pawn structure: doubled / isolated
# ---------------------------------------------------------------------------


def test_doubled_pawns_detected():
    # Two white pawns on e-file
    board = chess.Board("8/8/8/8/4P3/4P3/8/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "e" in ctx["doubled_files_white"]


def test_no_doubled_pawns_start():
    ctx = compute_position_context(chess.Board())
    assert ctx["doubled_files_white"] == []


def test_isolated_pawn_detected():
    # White pawn on a2 with no white pawns on b-file
    board = chess.Board("8/8/8/8/8/8/P7/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "a2" in ctx["isolated_pawns_white"]


def test_pawn_with_neighbor_not_isolated():
    # White pawns on a2 and b2 — a2 has neighbour on b
    board = chess.Board("8/8/8/8/8/8/PP6/4K2k w - - 0 1")
    ctx = compute_position_context(board)
    assert "a2" not in ctx["isolated_pawns_white"]


# ---------------------------------------------------------------------------
# Open / half-open files
# ---------------------------------------------------------------------------


def test_no_open_files_at_start():
    ctx = compute_position_context(chess.Board())
    assert ctx["open_files"] == []


def test_open_file_after_pawns_traded():
    # Remove e-file pawns from starting position
    board = chess.Board()
    board.remove_piece_at(chess.E2)
    board.remove_piece_at(chess.E7)
    ctx = compute_position_context(board)
    assert "e" in ctx["open_files"]


def test_half_open_file():
    # Remove white e-pawn only: half-open for white
    board = chess.Board()
    board.remove_piece_at(chess.E2)
    ctx = compute_position_context(board)
    assert "e" in ctx["half_open_white"]
    assert "e" not in ctx["half_open_black"]


# ---------------------------------------------------------------------------
# King safety
# ---------------------------------------------------------------------------


def test_starting_king_shield():
    # White king on e1: shield squares are d2, e2, f2 — all have pawns
    ctx = compute_position_context(chess.Board())
    assert ctx["king_shield_white"] == 3


def test_castled_king_shield():
    # White king on g1 with f2, g2, h2 pawns intact
    board = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQ - 0 5")
    board.push_san("O-O")
    ctx = compute_position_context(board)
    # King on g1: shield squares g2, f2, h2 — all pawns present
    assert ctx["king_shield_white"] >= 2


def test_open_file_near_king_detected():
    # White king on g1, remove g-pawn → open g-file near king
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQ - 0 5")
    board.push_san("O-O")
    board.remove_piece_at(chess.G2)
    board.remove_piece_at(chess.G7)  # make g-file open
    ctx = compute_position_context(board)
    assert "g" in ctx["open_near_king_white"]


# ---------------------------------------------------------------------------
# Piece mobility
# ---------------------------------------------------------------------------


def test_mobility_at_start():
    ctx = compute_position_context(chess.Board())
    # Both sides should have identical mobility from start position
    assert ctx["mobility_white"] == ctx["mobility_black"]
    # Knights + bishops + rooks + queens — rooks/queens blocked; knights have some squares
    assert ctx["mobility_white"] > 0


def test_more_open_position_more_mobility():
    # After e4 e5 Nf3 — more pieces developed, more mobility
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    ctx = compute_position_context(board)
    # White has knight out + bishop diagonal — more mobility than start
    assert ctx["mobility_white"] > 0


# ---------------------------------------------------------------------------
# format_position_context integration
# ---------------------------------------------------------------------------


def test_format_position_context_sections():
    from tutor.prompts import format_position_context

    ctx = compute_position_context(chess.Board())
    text = format_position_context(ctx)
    assert "## Position Context" in text
    assert "Phase:" in text
    assert "Material" in text
    assert "Pawn structure" in text
    assert "King safety" in text
    assert "Piece mobility" in text


def test_format_position_context_empty():
    from tutor.prompts import format_position_context

    assert format_position_context({}) == ""


def test_format_user_prompt_includes_position_context():
    from tutor.prompts import format_user_prompt

    ctx = compute_position_context(chess.Board())
    board = chess.Board()
    prompt = format_user_prompt(
        board_ascii_str=str(board),
        san="e4",
        classification="Best",
        eval_str="+0.30",
        fen=chess.STARTING_FEN,
        position_context=ctx,
    )
    assert "## Position Context" in prompt
    assert "## Position" in prompt
    assert "Phase:" in prompt
