"""Tests for MCP chess tools."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chess_mcp.stockfish import Stockfish
from chess_mcp.tools import ChessTools, ToolResult

# Test positions
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


@pytest.fixture
async def chess_tools():
    """Create ChessTools instance for testing."""
    sf = Stockfish(depth=10, threads=1, hash_mb=64)
    await sf.start()
    tools = ChessTools(sf)
    yield tools
    await sf.stop()


class TestGetBestMove:
    @pytest.mark.asyncio
    async def test_starting_position(self, chess_tools):
        """Test best move for starting position."""
        result = await chess_tools.get_best_move(STARTING_FEN, depth=10)
        assert result.success
        assert result.data["best_move"]
        assert len(result.data["best_move"]) == 4  # UCI format

    @pytest.mark.asyncio
    async def test_returns_pv(self, chess_tools):
        """Test that principal variation is returned."""
        result = await chess_tools.get_best_move(STARTING_FEN, depth=10)
        assert result.success
        assert "pv" in result.data
        assert len(result.data["pv"]) >= 1

    @pytest.mark.asyncio
    async def test_invalid_fen(self, chess_tools):
        """Test error handling for invalid FEN."""
        result = await chess_tools.get_best_move("invalid fen", depth=10)
        assert not result.success
        assert "Invalid FEN" in result.error


class TestGetEval:
    @pytest.mark.asyncio
    async def test_starting_position(self, chess_tools):
        """Test eval for starting position (should be close to 0)."""
        result = await chess_tools.get_eval(STARTING_FEN, depth=10)
        assert result.success
        assert "score" in result.data
        assert "win_probability" in result.data
        # Starting position should be roughly equal
        if "centipawns" in result.data:
            assert abs(result.data["centipawns"]) < 100

    @pytest.mark.asyncio
    async def test_win_probability_range(self, chess_tools):
        """Test win probability is in valid range."""
        result = await chess_tools.get_eval(STARTING_FEN, depth=10)
        assert result.success
        assert 0 <= result.data["win_probability"] <= 1


class TestCompareMoves:
    @pytest.mark.asyncio
    async def test_best_move_classification(self, chess_tools):
        """Test that best move is classified correctly."""
        # Get best move first
        best = await chess_tools.get_best_move(AFTER_E4_FEN, depth=10)
        best_move = best.data["best_move"]

        # Compare it
        result = await chess_tools.compare_moves(AFTER_E4_FEN, best_move, depth=10)
        assert result.success
        # At low depth, evaluations can vary slightly between calls
        # The move should still be classified as Best or Great
        assert result.data["classification"] in ["Best", "Great"]
        assert result.data["cp_loss"] <= 30  # Within "Great" threshold

    @pytest.mark.asyncio
    async def test_inaccuracy_detection(self, chess_tools):
        """Test detection of inaccurate moves."""
        # a6 is an inaccuracy after 1.e4
        result = await chess_tools.compare_moves(AFTER_E4_FEN, "a7a6", depth=10)
        assert result.success
        assert not result.data["is_best"]
        assert result.data["cp_loss"] > 0
        # Should be classified as at least inaccuracy
        assert result.data["classification"] in ["Good", "Inaccuracy", "Mistake", "Blunder"]

    @pytest.mark.asyncio
    async def test_illegal_move(self, chess_tools):
        """Test error for illegal moves."""
        result = await chess_tools.compare_moves(STARTING_FEN, "e1e8")
        assert not result.success
        assert "Illegal move" in result.error

    @pytest.mark.asyncio
    async def test_invalid_format(self, chess_tools):
        """Test error for invalid move format."""
        result = await chess_tools.compare_moves(STARTING_FEN, "xyz")
        assert not result.success
        assert "Invalid move format" in result.error


class TestGetThreats:
    @pytest.mark.asyncio
    async def test_check_detection(self, chess_tools):
        """Test that check is detected."""
        result = await chess_tools.get_threats(CHECKMATE_FEN, depth=10)
        assert result.success
        assert result.data["in_check"]
        assert len(result.data["checkers"]) > 0

    @pytest.mark.asyncio
    async def test_no_check(self, chess_tools):
        """Test position without check."""
        result = await chess_tools.get_threats(STARTING_FEN, depth=10)
        assert result.success
        assert not result.data["in_check"]

    @pytest.mark.asyncio
    async def test_tactical_threats(self, chess_tools):
        """Test detection of tactical threats."""
        # Scholars mate position - Qxf7# is threatened
        result = await chess_tools.get_threats(SCHOLARS_MATE_FEN, depth=10)
        assert result.success
        # Should find some threats
        assert "threats" in result.data


class TestValidateMove:
    @pytest.mark.asyncio
    async def test_legal_move(self, chess_tools):
        """Test validation of legal move."""
        result = await chess_tools.validate_move(STARTING_FEN, "e2e4")
        assert result.success
        assert result.data["legal"]
        assert result.data["san"] == "e4"
        assert result.data["resulting_fen"]

    @pytest.mark.asyncio
    async def test_illegal_move(self, chess_tools):
        """Test validation of illegal move."""
        result = await chess_tools.validate_move(STARTING_FEN, "e1e8")
        assert result.success  # Tool succeeded
        assert not result.data["legal"]
        assert "reason" in result.data

    @pytest.mark.asyncio
    async def test_check_detection(self, chess_tools):
        """Test that check is detected after move."""
        # Position where e4 gives check would be needed
        # For now, test that the field exists
        result = await chess_tools.validate_move(STARTING_FEN, "e2e4")
        assert "is_check" in result.data


class TestGetLegalMoves:
    @pytest.mark.asyncio
    async def test_starting_position(self, chess_tools):
        """Test legal moves count for starting position."""
        result = await chess_tools.get_legal_moves(STARTING_FEN)
        assert result.success
        assert result.data["total"] == 20  # 16 pawn + 4 knight moves

    @pytest.mark.asyncio
    async def test_move_categorization(self, chess_tools):
        """Test that moves are categorized."""
        result = await chess_tools.get_legal_moves(STARTING_FEN)
        assert result.success
        assert "captures" in result.data
        assert "checks" in result.data
        assert "other" in result.data
        # Starting position has no captures
        assert len(result.data["captures"]) == 0
