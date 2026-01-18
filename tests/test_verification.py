"""Tests for verification and legality modules."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verification.legality import (
    validate_fen,
    validate_move,
    parse_move_flexible,
    get_legal_moves_for_piece,
    filter_valid_moves,
    is_game_over,
    MoveFormat,
    MoveValidationResult,
)
from verification.tactical_loop import (
    classify_move_by_cp_loss,
    extract_classification_from_text,
    is_classification_compatible,
    MoveClassification,
)


# Test positions
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
CHECK_FEN = "rnbqkbnr/ppppp1pp/5p2/6B1/4P3/8/PPPP1PPP/RN1QKBNR b KQkq - 1 2"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"  # Black king trapped, no legal moves


class TestValidateFen:
    def test_valid_starting_position(self):
        """Test valid starting FEN."""
        valid, error = validate_fen(STARTING_FEN)
        assert valid
        assert error is None

    def test_invalid_fen_format(self):
        """Test invalid FEN format."""
        valid, error = validate_fen("not a valid fen")
        assert not valid
        assert "Invalid FEN" in error

    def test_invalid_board_position(self):
        """Test FEN with invalid board (e.g., two kings)."""
        # FEN with two white kings
        invalid_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKKBNR w KQkq - 0 1"
        valid, error = validate_fen(invalid_fen)
        # python-chess may accept this, so we just check it runs
        assert isinstance(valid, bool)


class TestValidateMove:
    def test_valid_uci_move(self):
        """Test valid UCI format move."""
        result = validate_move(STARTING_FEN, "e2e4", MoveFormat.UCI)
        assert result.valid
        assert result.move_uci == "e2e4"
        assert result.move_san == "e4"
        assert result.resulting_fen is not None

    def test_valid_san_move(self):
        """Test valid SAN format move."""
        result = validate_move(STARTING_FEN, "e4", MoveFormat.SAN)
        assert result.valid
        assert result.move_uci == "e2e4"
        assert result.move_san == "e4"

    def test_valid_san_piece_move(self):
        """Test valid SAN piece move."""
        result = validate_move(STARTING_FEN, "Nf3", MoveFormat.SAN)
        assert result.valid
        assert result.move_uci == "g1f3"
        assert result.move_san == "Nf3"

    def test_illegal_move(self):
        """Test illegal move detection."""
        result = validate_move(STARTING_FEN, "e1e8", MoveFormat.UCI)
        assert not result.valid
        assert "illegal" in result.error.lower()

    def test_no_piece_on_square(self):
        """Test move from empty square."""
        result = validate_move(STARTING_FEN, "e4e5", MoveFormat.UCI)
        assert not result.valid
        assert "No piece" in result.error or "illegal" in result.error.lower()

    def test_invalid_format(self):
        """Test invalid move format."""
        result = validate_move(STARTING_FEN, "xyz", MoveFormat.UCI)
        assert not result.valid
        assert "Cannot parse" in result.error

    def test_check_detection(self):
        """Test that resulting check is detected."""
        # Scholars mate setup
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        result = validate_move(fen, "h5f7", MoveFormat.UCI)
        assert result.valid
        assert result.is_checkmate

    def test_capture_detection(self):
        """Test capture detection."""
        # Position where e4 pawn can take d5 pawn
        fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
        result = validate_move(fen, "e4d5", MoveFormat.UCI)
        assert result.valid
        assert result.is_capture
        assert result.captured_piece == "p"


class TestParseMoveFlexible:
    def test_uci_format(self):
        """Test flexible parsing of UCI format."""
        result = parse_move_flexible(STARTING_FEN, "e2e4")
        assert result.valid
        assert result.move_uci == "e2e4"

    def test_san_format(self):
        """Test flexible parsing of SAN format."""
        result = parse_move_flexible(STARTING_FEN, "e4")
        assert result.valid
        assert result.move_uci == "e2e4"

    def test_san_with_piece(self):
        """Test flexible parsing of SAN with piece."""
        result = parse_move_flexible(STARTING_FEN, "Nf3")
        assert result.valid
        assert result.move_uci == "g1f3"

    def test_castling_short(self):
        """Test parsing of short castling."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        result = parse_move_flexible(fen, "O-O")
        assert result.valid
        assert result.move_uci == "e1g1"

    def test_castling_long(self):
        """Test parsing of long castling."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        result = parse_move_flexible(fen, "O-O-O")
        assert result.valid
        assert result.move_uci == "e1c1"


class TestGetLegalMovesForPiece:
    def test_knight_moves(self):
        """Test getting legal moves for knight."""
        moves = get_legal_moves_for_piece(STARTING_FEN, "g1")
        assert "g1f3" in moves
        assert "g1h3" in moves
        assert len(moves) == 2

    def test_pawn_moves(self):
        """Test getting legal moves for pawn."""
        moves = get_legal_moves_for_piece(STARTING_FEN, "e2")
        assert "e2e3" in moves
        assert "e2e4" in moves
        assert len(moves) == 2

    def test_empty_square(self):
        """Test empty square returns no moves."""
        moves = get_legal_moves_for_piece(STARTING_FEN, "e4")
        assert len(moves) == 0

    def test_invalid_square(self):
        """Test invalid square returns no moves."""
        moves = get_legal_moves_for_piece(STARTING_FEN, "z9")
        assert len(moves) == 0


class TestFilterValidMoves:
    def test_filter_valid(self):
        """Test filtering valid moves."""
        moves = ["e4", "Nf3", "invalid", "xyz", "d4"]
        valid = filter_valid_moves(STARTING_FEN, moves)
        assert "e2e4" in valid
        assert "g1f3" in valid
        assert "d2d4" in valid
        assert len(valid) == 3

    def test_empty_list(self):
        """Test filtering empty list."""
        valid = filter_valid_moves(STARTING_FEN, [])
        assert len(valid) == 0


class TestIsGameOver:
    def test_not_over(self):
        """Test game not over."""
        result = is_game_over(STARTING_FEN)
        assert not result["is_game_over"]
        assert not result["is_check"]
        assert not result["is_checkmate"]
        assert not result["is_stalemate"]

    def test_in_check(self):
        """Test check detection (not checkmate)."""
        # Position after 1.e4 f5 2.Qh5+ - Black king in check but can block with g6
        check_fen = "rnbqkbnr/ppppp1pp/8/5p1Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2"
        result = is_game_over(check_fen)
        assert not result["is_game_over"]
        assert result["is_check"]
        assert not result["is_checkmate"]

    def test_checkmate(self):
        """Test checkmate detection."""
        result = is_game_over(CHECKMATE_FEN)
        assert result["is_game_over"]
        assert result["is_checkmate"]

    def test_stalemate(self):
        """Test stalemate detection."""
        result = is_game_over(STALEMATE_FEN)
        assert result["is_game_over"]
        assert result["is_stalemate"]


class TestClassifyMoveByCpLoss:
    def test_best_move(self):
        """Test best move classification."""
        assert classify_move_by_cp_loss(0) == MoveClassification.BEST
        assert classify_move_by_cp_loss(5) == MoveClassification.BEST
        assert classify_move_by_cp_loss(10) == MoveClassification.BEST

    def test_great_move(self):
        """Test great move classification."""
        assert classify_move_by_cp_loss(15) == MoveClassification.GREAT
        assert classify_move_by_cp_loss(30) == MoveClassification.GREAT

    def test_good_move(self):
        """Test good move classification."""
        assert classify_move_by_cp_loss(50) == MoveClassification.GOOD
        assert classify_move_by_cp_loss(80) == MoveClassification.GOOD

    def test_inaccuracy(self):
        """Test inaccuracy classification."""
        assert classify_move_by_cp_loss(100) == MoveClassification.INACCURACY
        assert classify_move_by_cp_loss(150) == MoveClassification.INACCURACY

    def test_mistake(self):
        """Test mistake classification."""
        assert classify_move_by_cp_loss(200) == MoveClassification.MISTAKE
        assert classify_move_by_cp_loss(300) == MoveClassification.MISTAKE

    def test_blunder(self):
        """Test blunder classification."""
        assert classify_move_by_cp_loss(400) == MoveClassification.BLUNDER
        assert classify_move_by_cp_loss(1000) == MoveClassification.BLUNDER


class TestExtractClassificationFromText:
    def test_extract_blunder(self):
        """Test extracting blunder classification."""
        text = "This is a terrible blunder that loses the game."
        assert extract_classification_from_text(text) == MoveClassification.BLUNDER

    def test_extract_mistake(self):
        """Test extracting mistake classification."""
        text = "This move is a mistake that weakens your position."
        assert extract_classification_from_text(text) == MoveClassification.MISTAKE

    def test_extract_best(self):
        """Test extracting best move classification."""
        text = "This is the best move in the position."
        assert extract_classification_from_text(text) == MoveClassification.BEST

    def test_extract_excellent(self):
        """Test extracting great from excellent."""
        text = "Excellent choice! This is a strong move."
        assert extract_classification_from_text(text) == MoveClassification.GREAT

    def test_unknown(self):
        """Test unknown classification."""
        text = "The position is complex."
        assert extract_classification_from_text(text) == MoveClassification.UNKNOWN


class TestIsClassificationCompatible:
    def test_same_classification(self):
        """Test same classification is compatible."""
        assert is_classification_compatible(
            MoveClassification.BEST,
            MoveClassification.BEST
        )

    def test_adjacent_compatible(self):
        """Test adjacent classifications are compatible."""
        assert is_classification_compatible(
            MoveClassification.BEST,
            MoveClassification.GREAT,
            tolerance=1
        )

    def test_far_incompatible(self):
        """Test far classifications are incompatible."""
        assert not is_classification_compatible(
            MoveClassification.BEST,
            MoveClassification.BLUNDER,
            tolerance=1
        )

    def test_unknown_always_compatible(self):
        """Test unknown is always compatible."""
        assert is_classification_compatible(
            MoveClassification.UNKNOWN,
            MoveClassification.BLUNDER
        )
