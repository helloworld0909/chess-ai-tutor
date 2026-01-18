"""Tests for chess position representations."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chess_mcp.representations import (
    fen_to_ascii,
    fen_to_piece_squares,
    get_all_representations,
    format_for_llm,
    create_training_context,
    render_board_svg,
)


# Test positions
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
ENDGAME_FEN = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"


class TestFenToAscii:
    def test_starting_position(self):
        """Test ASCII board for starting position."""
        ascii_board = fen_to_ascii(STARTING_FEN)

        # Check ranks are present
        assert "8 |" in ascii_board
        assert "1 |" in ascii_board

        # Check files label
        assert "a b c d e f g h" in ascii_board

        # Check pieces
        assert "r n b q k b n r" in ascii_board  # Black pieces
        assert "R N B Q K B N R" in ascii_board  # White pieces
        assert "p p p p p p p p" in ascii_board  # Black pawns
        assert "P P P P P P P P" in ascii_board  # White pawns

    def test_after_e4(self):
        """Test ASCII board after 1.e4."""
        ascii_board = fen_to_ascii(AFTER_E4_FEN)

        # Pawn should be on e4
        lines = ascii_board.split("\n")
        rank4 = [l for l in lines if l.startswith("4 |")][0]
        assert "P" in rank4

    def test_flipped_board(self):
        """Test flipped board (Black's perspective)."""
        normal = fen_to_ascii(STARTING_FEN, flip=False)
        flipped = fen_to_ascii(STARTING_FEN, flip=True)

        # Normal has rank 8 at top
        normal_lines = normal.split("\n")
        assert normal_lines[0].startswith("8")

        # Flipped has rank 1 at top
        flipped_lines = flipped.split("\n")
        assert flipped_lines[0].startswith("1")

    def test_empty_squares(self):
        """Test that empty squares show dots."""
        ascii_board = fen_to_ascii(STARTING_FEN)
        # Rank 5 should be all empty
        lines = ascii_board.split("\n")
        rank5 = [l for l in lines if l.startswith("5 |")][0]
        assert ". . . . . . . ." in rank5


class TestFenToPieceSquares:
    def test_starting_position(self):
        """Test piece-square list for starting position."""
        pieces = fen_to_piece_squares(STARTING_FEN)

        assert "white" in pieces
        assert "black" in pieces

        # Should have 16 pieces each
        assert len(pieces["white"]) == 16
        assert len(pieces["black"]) == 16

    def test_piece_format(self):
        """Test piece info format."""
        pieces = fen_to_piece_squares(STARTING_FEN)

        # Check white king
        white_king = [p for p in pieces["white"] if p["piece"] == "K"][0]
        assert white_king["square"] == "e1"
        assert white_king["name"] == "king"

        # Check black queen
        black_queen = [p for p in pieces["black"] if p["piece"] == "q"][0]
        assert black_queen["square"] == "d8"
        assert black_queen["name"] == "queen"

    def test_piece_order(self):
        """Test pieces are sorted by value (K, Q, R, B, N, P)."""
        pieces = fen_to_piece_squares(STARTING_FEN)

        white_types = [p["name"] for p in pieces["white"]]
        assert white_types[0] == "king"
        assert white_types[1] == "queen"

    def test_endgame_position(self):
        """Test piece count in endgame."""
        pieces = fen_to_piece_squares(ENDGAME_FEN)

        assert len(pieces["white"]) == 2  # K + P
        assert len(pieces["black"]) == 1  # K


class TestGetAllRepresentations:
    def test_all_fields_present(self):
        """Test all representation fields are present."""
        reps = get_all_representations(STARTING_FEN)

        assert reps.fen == STARTING_FEN
        assert reps.ascii_board
        assert reps.piece_squares
        assert reps.side_to_move == "white"
        assert reps.castling_rights == "KQkq"
        assert reps.en_passant is None
        assert reps.halfmove_clock == 0
        assert reps.fullmove_number == 1

    def test_after_e4(self):
        """Test representations after 1.e4."""
        reps = get_all_representations(AFTER_E4_FEN)

        assert reps.side_to_move == "black"
        assert reps.en_passant == "e3"

    def test_endgame_no_castling(self):
        """Test endgame with no castling rights."""
        reps = get_all_representations(ENDGAME_FEN)

        assert reps.castling_rights == "-"


class TestFormatForLlm:
    def test_contains_fen(self):
        """Test LLM format contains FEN."""
        formatted = format_for_llm(STARTING_FEN)
        assert STARTING_FEN in formatted

    def test_contains_board(self):
        """Test LLM format contains board."""
        formatted = format_for_llm(STARTING_FEN)
        assert "Board:" in formatted
        assert "r n b q k b n r" in formatted

    def test_contains_pieces(self):
        """Test LLM format contains piece list."""
        formatted = format_for_llm(STARTING_FEN)
        assert "Pieces:" in formatted
        assert "White:" in formatted
        assert "Black:" in formatted

    def test_contains_side_to_move(self):
        """Test LLM format contains side to move."""
        formatted = format_for_llm(STARTING_FEN)
        assert "Side to move" in formatted
        assert "White" in formatted

    def test_castling_rights(self):
        """Test LLM format shows castling rights."""
        formatted = format_for_llm(STARTING_FEN)
        assert "Castling" in formatted
        assert "kingside" in formatted.lower() or "queenside" in formatted.lower()

    def test_en_passant(self):
        """Test LLM format shows en passant square."""
        formatted = format_for_llm(AFTER_E4_FEN)
        assert "En passant" in formatted
        assert "e3" in formatted

    def test_image_placeholder(self):
        """Test optional image placeholder."""
        formatted = format_for_llm(STARTING_FEN, include_image_placeholder=True)
        assert "[IMAGE" in formatted


class TestCreateTrainingContext:
    def test_basic_context(self):
        """Test basic training context."""
        context = create_training_context(STARTING_FEN)

        assert context["fen"] == STARTING_FEN
        assert context["ascii_board"]
        assert context["piece_squares"]
        assert context["side_to_move"] == "white"

    def test_with_move(self):
        """Test context with user move."""
        context = create_training_context(STARTING_FEN, user_move="e2e4")

        assert context["user_move"] == "e2e4"
        assert context["user_move_san"] == "e4"
        assert not context["is_capture"]
        assert not context["is_check"]

    def test_with_question(self):
        """Test context with question."""
        context = create_training_context(
            STARTING_FEN,
            question="What is the best move here?"
        )

        assert context["question"] == "What is the best move here?"


class TestRenderBoardSvg:
    def test_basic_svg(self):
        """Test basic SVG rendering."""
        svg = render_board_svg(STARTING_FEN)

        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "viewBox" in svg

    def test_with_size(self):
        """Test SVG with custom size."""
        svg = render_board_svg(STARTING_FEN, size=200)

        assert 'width="200"' in svg or "200" in svg

    def test_with_last_move(self):
        """Test SVG with highlighted last move."""
        svg = render_board_svg(AFTER_E4_FEN, last_move="e2e4")

        # Should still produce valid SVG
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_flipped(self):
        """Test flipped SVG."""
        svg = render_board_svg(STARTING_FEN, flip=True)

        assert svg.startswith("<svg")
        assert "</svg>" in svg
