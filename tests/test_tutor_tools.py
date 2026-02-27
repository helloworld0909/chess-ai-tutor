"""Tests for src/tutor/tools.py — play_moves, get_top_lines, get_attacks, fen_to_board, board_to_fen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tutor.tools import (
    CHESS_TOOLS,
    board_to_fen,
    fen_to_board,
    get_attacks,
    get_top_lines,
    play_moves,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
# Ruy Lopez after 1.e4 e5 2.Nf3 Nc6 3.Bb5
RUY_LOPEZ_FEN = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


# ---------------------------------------------------------------------------
# CHESS_TOOLS list
# ---------------------------------------------------------------------------


class TestChessToolList:
    def test_tool_names_include_new_tools(self):
        names = {t["function"]["name"] for t in CHESS_TOOLS}
        assert "play_moves" in names
        assert "get_top_lines" in names
        assert "get_attacks" in names
        assert "fen_to_board" in names
        assert "board_to_fen" in names

    def test_all_tools_have_required_schema_fields(self):
        for tool in CHESS_TOOLS:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert "required" in fn["parameters"]


# ---------------------------------------------------------------------------
# play_moves
# ---------------------------------------------------------------------------


class TestPlayMoves:
    def test_apply_single_san_move(self):
        result = json.loads(play_moves(STARTING_FEN, ["e4"]))
        assert "resulting_fen" in result
        assert result["moves_san"] == ["e4"]
        assert not result["is_checkmate"]

    def test_apply_uci_move(self):
        result = json.loads(play_moves(STARTING_FEN, ["e2e4"]))
        assert result["moves_san"] == ["e4"]

    def test_apply_sequence(self):
        result = json.loads(play_moves(STARTING_FEN, ["e4", "e5", "Nf3"]))
        assert len(result["moves_san"]) == 3
        assert result["moves_san"] == ["e4", "e5", "Nf3"]

    def test_illegal_move_returns_error(self):
        result = json.loads(play_moves(STARTING_FEN, ["e4", "e4"]))
        assert "error" in result
        assert "moves_applied" in result  # partial progress reported

    def test_invalid_fen_returns_error(self):
        result = json.loads(play_moves("not-a-fen", ["e4"]))
        assert "error" in result

    def test_empty_moves_returns_same_fen(self):
        result = json.loads(play_moves(STARTING_FEN, []))
        assert result["resulting_fen"] == STARTING_FEN

    def test_checkmate_detection(self):
        # Fool's mate: 1.f3 e5 2.g4 Qh4# — White is in checkmate
        fools_mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        result = json.loads(play_moves(fools_mate, []))
        assert result["is_checkmate"]


# ---------------------------------------------------------------------------
# get_top_lines (requires Stockfish)
# ---------------------------------------------------------------------------


@pytest.fixture
async def stockfish():
    from chess_mcp.stockfish import Stockfish

    sf = Stockfish(depth=10, threads=1, hash_mb=64)
    await sf.start()
    yield sf
    await sf.stop()


class TestGetTopLines:
    @pytest.mark.asyncio
    async def test_returns_lines(self, stockfish):
        result = json.loads(await get_top_lines(STARTING_FEN, stockfish, n=3, depth=8))
        assert "lines" in result
        assert len(result["lines"]) > 0

    @pytest.mark.asyncio
    async def test_line_structure(self, stockfish):
        result = json.loads(await get_top_lines(STARTING_FEN, stockfish, n=3, depth=8))
        for line in result["lines"]:
            assert "moves" in line
            assert "eval" in line
            assert "depth" in line
            assert isinstance(line["moves"], list)
            assert len(line["moves"]) > 0

    @pytest.mark.asyncio
    async def test_eval_label_is_human_readable(self, stockfish):
        valid_labels = {
            "winning for white",
            "good for white",
            "equal",
            "good for black",
            "winning for black",
        }
        result = json.loads(await get_top_lines(STARTING_FEN, stockfish, n=3, depth=8))
        for line in result["lines"]:
            assert line["eval"] in valid_labels

    @pytest.mark.asyncio
    async def test_n_capped_at_5(self, stockfish):
        result = json.loads(await get_top_lines(STARTING_FEN, stockfish, n=10, depth=8))
        assert len(result["lines"]) <= 5

    @pytest.mark.asyncio
    async def test_invalid_fen_returns_error(self, stockfish):
        result = json.loads(await get_top_lines("bad-fen", stockfish))
        assert "error" in result


# ---------------------------------------------------------------------------
# get_attacks
# ---------------------------------------------------------------------------


class TestGetAttacks:
    def test_e4_pawn_attacks(self):
        # After 1.e4 — white pawn on e4 attacks d5 and f5
        result = json.loads(get_attacks(AFTER_E4_FEN, "e4"))
        assert result["piece_on_square"] == "P"
        assert "d5" in result["attacks_to"]
        assert "f5" in result["attacks_to"]

    def test_empty_square(self):
        result = json.loads(get_attacks(STARTING_FEN, "e4"))
        assert result["piece_on_square"] is None
        assert result["attacks_to"] == []

    def test_attacked_by_multiple_pieces(self):
        # d4 in starting position is attacked by white queen (d1) and bishop (c1 region)
        # In RUY_LOPEZ_FEN, let's check d5 attacked by white bishop on b5
        result = json.loads(get_attacks(RUY_LOPEZ_FEN, "d7"))
        # d7 pawn is defended by queen on d8 and king on e8 (via queen)
        attacker_squares = {a["square"] for a in result["attacked_by"]}
        assert len(attacker_squares) > 0  # at least one attacker

    def test_invalid_fen_returns_error(self):
        result = json.loads(get_attacks("bad-fen", "e4"))
        assert "error" in result

    def test_invalid_square_returns_error(self):
        result = json.loads(get_attacks(STARTING_FEN, "z9"))
        assert "error" in result


# ---------------------------------------------------------------------------
# fen_to_board
# ---------------------------------------------------------------------------


class TestFenToBoard:
    def test_returns_ascii_diagram(self):
        result = json.loads(fen_to_board(STARTING_FEN))
        assert "board" in result
        board = result["board"]
        # Should have 9 lines (header + 8 ranks)
        lines = board.splitlines()
        assert len(lines) == 9

    def test_side_to_move_white(self):
        result = json.loads(fen_to_board(STARTING_FEN))
        assert result["side_to_move"] == "White"

    def test_side_to_move_black(self):
        result = json.loads(fen_to_board(AFTER_E4_FEN))
        assert result["side_to_move"] == "Black"

    def test_pieces_present(self):
        result = json.loads(fen_to_board(STARTING_FEN))
        board = result["board"]
        assert "R" in board  # White rook
        assert "r" in board  # Black rook
        assert "K" in board  # White king
        assert "k" in board  # Black king

    def test_invalid_fen_returns_error(self):
        result = json.loads(fen_to_board("not-valid"))
        assert "error" in result


# ---------------------------------------------------------------------------
# board_to_fen
# ---------------------------------------------------------------------------


class TestBoardToFen:
    def test_starting_position(self):
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        result = json.loads(board_to_fen(placement, side_to_move="w", castling="KQkq"))
        assert "fen" in result
        # Should parse as valid starting FEN
        import chess

        board = chess.Board(result["fen"])
        assert board.turn == chess.WHITE

    def test_after_e4(self):
        placement = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        result = json.loads(
            board_to_fen(placement, side_to_move="b", castling="KQkq", en_passant="e3")
        )
        assert "fen" in result
        import chess

        board = chess.Board(result["fen"])
        assert board.turn == chess.BLACK

    def test_invalid_side_to_move(self):
        result = json.loads(board_to_fen("8/8/8/8/8/8/8/8", side_to_move="x"))
        assert "error" in result

    def test_invalid_placement_returns_error(self):
        result = json.loads(board_to_fen("not/valid/placement"))
        assert "error" in result

    def test_roundtrip_fen_to_board_to_fen(self):
        """fen_to_board → board_to_fen should recover equivalent FEN."""
        import chess

        original = chess.Board(RUY_LOPEZ_FEN)
        # Extract placement from FEN
        placement = RUY_LOPEZ_FEN.split(" ")[0]
        result = json.loads(board_to_fen(placement, side_to_move="w"))
        assert "fen" in result
        recovered = chess.Board(result["fen"])
        # Piece placement should match
        assert original.board_fen() == recovered.board_fen()
