"""Tests for data/pipeline/generate_lines.py."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "pipeline"))

from generate_lines import (
    LABEL_WINNING_BLACK,
    _extract_positions_from_transcript,
    cp_to_label,
    generate_lines_for_position,
)

# ---------------------------------------------------------------------------
# cp_to_label
# ---------------------------------------------------------------------------


class TestCpToLabel:
    def test_winning_for_white(self):
        assert cp_to_label(350) == "winning for white"

    def test_good_for_white(self):
        assert cp_to_label(200) == "good for white"

    def test_equal_positive_edge(self):
        assert cp_to_label(100) == "equal"

    def test_equal_zero(self):
        assert cp_to_label(0) == "equal"

    def test_equal_negative_edge(self):
        assert cp_to_label(-100) == "equal"

    def test_good_for_black(self):
        assert cp_to_label(-200) == "good for black"

    def test_winning_for_black(self):
        assert cp_to_label(-350) == "winning for black"

    def test_mate_for_white(self):
        assert cp_to_label(None, is_mate=True, mate_value=3) == "winning for white"

    def test_mate_for_black(self):
        assert cp_to_label(None, is_mate=True, mate_value=-2) == "winning for black"

    def test_none_cp_returns_equal(self):
        assert cp_to_label(None) == "equal"

    def test_boundary_300(self):
        assert cp_to_label(300) == "winning for white"

    def test_boundary_299(self):
        assert cp_to_label(299) == "good for white"

    def test_boundary_100(self):
        assert cp_to_label(100) == "equal"

    def test_boundary_101(self):
        assert cp_to_label(101) == "good for white"

    def test_boundary_minus_100(self):
        assert cp_to_label(-100) == "equal"

    def test_boundary_minus_101(self):
        assert cp_to_label(-101) == "good for black"

    def test_boundary_minus_300(self):
        assert cp_to_label(-300) == "good for black"

    def test_boundary_minus_301(self):
        assert cp_to_label(-301) == "winning for black"


# ---------------------------------------------------------------------------
# _extract_positions_from_transcript
# ---------------------------------------------------------------------------


class TestExtractPositions:
    def _make_rng(self, choice_index: int = 0):
        import random

        rng = random.Random(42)
        # Override choice to be deterministic in tests
        original_choice = rng.choice
        call_count = [0]

        def fixed_choice(seq):
            idx = min(choice_index, len(seq) - 1)
            return seq[idx]

        rng.choice = fixed_choice
        return rng

    def test_short_game_returns_none(self):
        # Only 6 moves (12 plies) — below MIN_MOVE_NUMBER * 2 = 16
        transcript = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6"
        import random

        result = _extract_positions_from_transcript(
            transcript, 1500, 1500, "amateur", random.Random(42)
        )
        assert result is None

    def test_valid_game_returns_position(self):
        # Standard opening moves in UCI (Ruy Lopez, all moves verified legal)
        transcript = (
            "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 "
            "e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 "
            "h2h3 c6a5 b3c2 c7c5 d2d4"
        )
        import random

        result = _extract_positions_from_transcript(
            transcript, 1500, 1600, "amateur", random.Random(42)
        )
        assert result is not None
        assert "fen" in result
        assert "move_san" in result
        assert "move_uci" in result
        assert result["white_elo"] == 1500
        assert result["black_elo"] == 1600
        assert result["tier"] == "amateur"
        # FEN should be a valid chess position
        board = chess.Board(result["fen"])
        assert board is not None

    def test_invalid_move_returns_none(self):
        transcript = "e2e4 e7e5 g1f3 INVALID e7e5"
        import random

        result = _extract_positions_from_transcript(
            transcript, 1500, 1500, "amateur", random.Random(42)
        )
        assert result is None


# ---------------------------------------------------------------------------
# generate_lines_for_position
# ---------------------------------------------------------------------------


class TestGenerateLinesForPosition:
    def _make_mock_analysis(self, pvs: list[list[str]], scores: list[int]):
        """Create a mock Analysis object with the given PVs and cp scores."""
        from chess_mcp.stockfish import Analysis, AnalysisLine, Score, ScoreType

        lines = []
        for i, (pv, cp) in enumerate(zip(pvs, scores), 1):
            line = MagicMock(spec=AnalysisLine)
            line.pv = pv
            line.score = Score(ScoreType.CENTIPAWNS, cp)
            line.multipv = i
            lines.append(line)

        analysis = MagicMock(spec=Analysis)
        analysis.lines = lines
        return analysis

    @pytest.mark.asyncio
    async def test_basic_line_generation(self):
        """Lines are generated correctly from a starting position."""
        # Starting position, e2e4
        fen = chess.STARTING_FEN
        move_uci = "e2e4"
        move_san = "e4"

        mock_sf = AsyncMock()
        mock_sf.analyze = AsyncMock(
            return_value=self._make_mock_analysis(
                pvs=[["e7e5", "g1f3"], ["c7c5", "g1f3"], ["e7e6", "d2d4"]],
                scores=[15, 30, 20],
            )
        )

        lines = await generate_lines_for_position(mock_sf, fen, move_uci, move_san)

        assert len(lines) == 3
        assert lines[0].startswith("LINE 1:")
        assert "→" in lines[0]
        assert "| eval:" in lines[0]
        # cp=15 → equal (within -100 to +100)
        assert "equal" in lines[0]
        # cp=30 → equal
        assert "equal" in lines[1]

    @pytest.mark.asyncio
    async def test_winning_eval_label(self):
        """High cp score maps to 'winning for white'."""
        fen = chess.STARTING_FEN
        mock_sf = AsyncMock()
        mock_sf.analyze = AsyncMock(
            return_value=self._make_mock_analysis(
                pvs=[["e7e5", "g1f3"]],
                scores=[400],
            )
        )

        lines = await generate_lines_for_position(mock_sf, fen, "e2e4", "e4")
        assert "winning for white" in lines[0]

    @pytest.mark.asyncio
    async def test_invalid_move_returns_empty(self):
        """Invalid move_uci returns empty list."""
        mock_sf = AsyncMock()
        lines = await generate_lines_for_position(mock_sf, chess.STARTING_FEN, "z9z9", "??")
        assert lines == []

    @pytest.mark.asyncio
    async def test_empty_pv_skipped(self):
        """Lines with empty PV are skipped."""
        fen = chess.STARTING_FEN
        mock_sf = AsyncMock()
        mock_sf.analyze = AsyncMock(
            return_value=self._make_mock_analysis(
                pvs=[[], ["e7e5"]],
                scores=[0, 20],
            )
        )

        lines = await generate_lines_for_position(mock_sf, fen, "e2e4", "e4")
        # First line skipped (empty pv), second line present (keeps its original number)
        assert len(lines) == 1
        assert "LINE 2:" in lines[0]

    @pytest.mark.asyncio
    async def test_san_moves_in_output(self):
        """Output contains SAN moves, not UCI."""
        fen = chess.STARTING_FEN
        mock_sf = AsyncMock()
        mock_sf.analyze = AsyncMock(
            return_value=self._make_mock_analysis(
                pvs=[["e7e5", "g1f3"]],
                scores=[10],
            )
        )

        lines = await generate_lines_for_position(mock_sf, fen, "e2e4", "e4")
        assert len(lines) == 1
        # Should be SAN (e5, Nf3), not UCI (e7e5, g1f3)
        assert "e7e5" not in lines[0]
        assert "g1f3" not in lines[0]
        assert "e5" in lines[0]
        assert "Nf3" in lines[0]
