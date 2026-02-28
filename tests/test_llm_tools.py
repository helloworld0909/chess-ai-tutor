"""Tests for LLM tool definitions and handler."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chess_mcp.stockfish import Stockfish
from tutor.llm_tools import CHESS_TOOLS, ChessToolHandler

# Test positions
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


class TestChessToolSchemas:
    def test_tool_count(self):
        """Test that all expected tools are defined."""
        assert len(CHESS_TOOLS) == 5

    def test_tool_names(self):
        """Test tool names are correct."""
        names = {t["function"]["name"] for t in CHESS_TOOLS}
        expected = {"get_best_move", "get_eval", "analyze_move", "get_legal_moves", "validate_move"}
        assert names == expected

    def test_tool_schema_structure(self):
        """Test each tool has required schema fields."""
        for tool in CHESS_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
            assert "required" in func["parameters"]


@pytest.fixture
async def tool_handler():
    """Create ChessToolHandler instance for testing."""
    sf = Stockfish(depth=10, threads=1, hash_mb=64)
    await sf.start()
    handler = ChessToolHandler(sf)
    yield handler
    await sf.stop()


class TestChessToolHandler:
    @pytest.mark.asyncio
    async def test_get_best_move(self, tool_handler):
        """Test get_best_move tool."""
        result = await tool_handler.handle_tool_call(
            "get_best_move", {"fen": STARTING_FEN, "depth": 10}
        )
        data = json.loads(result)

        assert "best_move" in data
        assert len(data["best_move"]) >= 4  # UCI format
        assert "score" in data
        assert "win_probability" in data
        assert "principal_variation" in data

    @pytest.mark.asyncio
    async def test_get_eval(self, tool_handler):
        """Test get_eval tool."""
        result = await tool_handler.handle_tool_call("get_eval", {"fen": STARTING_FEN, "depth": 10})
        data = json.loads(result)

        assert "score" in data
        assert "win_probability" in data
        # Starting position should have centipawns (not mate)
        assert "centipawns" in data

    @pytest.mark.asyncio
    async def test_analyze_move_best(self, tool_handler):
        """Test analyze_move for a good move."""
        # First get the best move
        best_result = await tool_handler.handle_tool_call(
            "get_best_move", {"fen": AFTER_E4_FEN, "depth": 10}
        )
        best_move = json.loads(best_result)["best_move"]

        # Analyze it
        result = await tool_handler.handle_tool_call(
            "analyze_move", {"fen": AFTER_E4_FEN, "move": best_move, "depth": 10}
        )
        data = json.loads(result)

        assert "classification" in data
        assert data["classification"] in ["Best", "Great"]
        assert "is_best" in data
        assert "centipawn_loss" in data

    @pytest.mark.asyncio
    async def test_analyze_move_bad(self, tool_handler):
        """Test analyze_move for a suboptimal move."""
        result = await tool_handler.handle_tool_call(
            "analyze_move", {"fen": AFTER_E4_FEN, "move": "a7a6", "depth": 10}
        )
        data = json.loads(result)

        assert "classification" in data
        assert not data["is_best"]
        assert data["centipawn_loss"] > 0

    @pytest.mark.asyncio
    async def test_get_legal_moves(self, tool_handler):
        """Test get_legal_moves tool."""
        result = await tool_handler.handle_tool_call("get_legal_moves", {"fen": STARTING_FEN})
        data = json.loads(result)

        assert data["total"] == 20  # 16 pawn + 4 knight moves
        assert "captures" in data
        assert "checks" in data
        assert "other" in data
        assert len(data["captures"]) == 0  # No captures from starting pos

    @pytest.mark.asyncio
    async def test_validate_move_legal(self, tool_handler):
        """Test validate_move for a legal move."""
        result = await tool_handler.handle_tool_call(
            "validate_move", {"fen": STARTING_FEN, "move": "e4"}
        )
        data = json.loads(result)

        assert data["legal"] is True
        assert data["move_san"] == "e4"
        assert data["move_uci"] == "e2e4"
        assert "resulting_fen" in data

    @pytest.mark.asyncio
    async def test_validate_move_illegal(self, tool_handler):
        """Test validate_move for an illegal move."""
        result = await tool_handler.handle_tool_call(
            "validate_move", {"fen": STARTING_FEN, "move": "e5"}
        )
        data = json.loads(result)

        assert data["legal"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_unknown_tool(self, tool_handler):
        """Test handling of unknown tool."""
        result = await tool_handler.handle_tool_call("unknown_tool", {"foo": "bar"})
        data = json.loads(result)

        assert "error" in data
        assert "Unknown tool" in data["error"]
