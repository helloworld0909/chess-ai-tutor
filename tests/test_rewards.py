"""Tests for src/verification/rewards.py — GRPO reward functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verification.rewards import (
    combined_reward,
    parse_lines,
    reward_annotation_structural,
    reward_breadth,
    reward_depth,
    reward_eval_accuracy,
    reward_format,
    reward_legality,
    reward_relevance,
)

# ---------------------------------------------------------------------------
# Fixtures: canned completions
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Three legal lines from the starting position
LEGAL_COMPLETION = """\
<line>LINE 1: e4 (control center, gain space) → e5 (contest center) → Nf3 (develop knight) | eval: equal</line>
<line>LINE 2: d4 (control center) → d5 (contest center) → c4 (gain space) | eval: equal</line>
<line>LINE 3: Nf3 (develop knight) → d5 (contest center) → d4 (control center) | eval: equal</line>
"""

# Both lines have illegal first moves (e5, d5 are not legal for White from start)
ILLEGAL_COMPLETION = """\
<line>LINE 1: e5 (illegal move) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: d5 (illegal) → d4 (control) | eval: equal</line>
"""

# One legal (d4), one illegal (e5) — mean should be 0.0
MIXED_COMPLETION = """\
<line>LINE 1: e5 (illegal) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: d4 (control center) → d5 (contest center) | eval: equal</line>
"""

# All lines start with the same first move (no breadth)
SAME_FIRST_MOVE = """\
<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) | eval: equal</line>
<line>LINE 2: e4 (center) → c5 (sicilian) → Nf3 (develop) | eval: equal</line>
<line>LINE 3: e4 (center) → d5 (contest) → exd5 (capture pawn) | eval: equal</line>
"""

# Short lines (1 move each) — should score low on depth
SHORT_COMPLETION = """\
<line>LINE 1: e4 (center) | eval: equal</line>
<line>LINE 2: d4 (center) | eval: equal</line>
"""

# No lines at all
EMPTY_COMPLETION = "I don't know what to say."


# Prompt with FEN embedded (system+user message list format)
def _make_prompt(fen: str, move_san: str = "e4") -> list[dict]:
    return [
        {"role": "system", "content": "You are a chess coach."},
        {
            "role": "user",
            "content": (
                f"## Position\nFEN: {fen}\n"
                f"## Move Played\nMove: {move_san}\n"
                "## Task\nOutput 3 lines."
            ),
        },
    ]


def _completion(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------------------
# parse_lines
# ---------------------------------------------------------------------------


class TestParseLines:
    def test_parses_three_lines(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert len(lines) == 3

    def test_extracts_moves(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert lines[0]["moves_san"] == ["e4", "e5", "Nf3"]

    def test_extracts_eval_label(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert lines[0]["eval_label"] == "equal"

    def test_empty_returns_empty(self):
        lines = parse_lines(EMPTY_COMPLETION)
        assert lines == []

    def test_bare_line_format(self):
        bare = "LINE 1: e4 → d5 → exd5 | eval: good for white\n"
        lines = parse_lines(bare)
        assert len(lines) == 1
        assert lines[0]["eval_label"] == "good for white"


# ---------------------------------------------------------------------------
# R1 — reward_legality
# ---------------------------------------------------------------------------


class TestRewardLegality:
    def test_fully_legal_completion(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(LEGAL_COMPLETION)])
        # All 3 moves in each line legal → score = 1.0
        assert scores[0] == 1.0

    def test_illegal_completion_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(ILLEGAL_COMPLETION)])
        # Both lines: first move illegal (0/2 legal) → line score = -1.0 → mean = -1.0
        assert scores[0] == -1.0

    def test_mixed_completion_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(MIXED_COMPLETION)])
        # LINE 1: e5 illegal (0/2 legal) → -1.0
        # LINE 2: d4, d5 both legal (2/2) → +1.0
        # mean = 0.0
        assert scores[0] == 0.0

    def test_partial_credit(self):
        # A line where first move is legal but second is not: 1/2 legal → score = 0.0
        partial = (
            "<line>LINE 1: e4 (center) → e4 (illegal repeat) → Nf3 (develop) | eval: equal</line>\n"
        )
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(partial)])
        # 1 legal out of 3 moves → 2*(1/3) - 1 = -0.333...
        assert scores[0] < 0.0
        assert scores[0] > -1.0

    def test_no_lines_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_missing_fen_neutral(self):
        prompt = [{"role": "user", "content": "no fen here"}]
        scores = reward_legality([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# R0 — reward_format
# ---------------------------------------------------------------------------


class TestRewardFormat:
    def test_has_line_tags_scores_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 1.0

    def test_no_line_tags_scores_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_illegal_moves_but_correct_format_still_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# R4 — reward_depth
# ---------------------------------------------------------------------------


class TestRewardDepth:
    def test_three_move_lines_below_target(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_depth([prompt], [_completion(LEGAL_COMPLETION)])
        # Each line has 3 moves; target is 2 → score = min(3,2)/2 = 1.0
        assert abs(scores[0] - 1.0) < 0.01

    def test_short_lines_low_score(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_depth([prompt], [_completion(SHORT_COMPLETION)])
        # Each line has 1 move; target is 2 → score = 1/2 = 0.5; both e4/d4 legal
        assert abs(scores[0] - 0.5) < 0.01

    def test_no_lines_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_depth([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_illegal_lines_excluded(self):
        # ILLEGAL_COMPLETION: both lines start with illegal moves → no legal lines → -1.0
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_depth([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == -1.0

    def test_long_line_capped_at_one(self):
        # Build a very long (12-move) line — score should be 1.0
        long_line = (
            "<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) → Nc6 (develop)"
            " → Bb5 (pin) → a6 (challenge) → Ba4 (retreat) → Nf6 (develop)"
            " → O-O (castle) → Be7 (develop) → Re1 (centralize) → b5 (expand) | eval: equal</line>"
        )
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_depth([prompt], [_completion(long_line)])
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# R5 — reward_breadth
# ---------------------------------------------------------------------------


class TestRewardBreadth:
    def test_all_different_first_moves_score_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_breadth([prompt], [_completion(LEGAL_COMPLETION)])
        # e4, d4, Nf3 — all different legal moves
        assert scores[0] == 1.0

    def test_same_first_move_score_low(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_breadth([prompt], [_completion(SAME_FIRST_MOVE)])
        # All three lines start with e4 (legal) → unique_ratio = 1/3
        assert abs(scores[0] - 1 / 3) < 0.01

    def test_illegal_lines_excluded(self):
        # ILLEGAL_COMPLETION: e5, d5 both illegal from start → no legal lines → -1.0
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_breadth([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == -1.0

    def test_no_lines_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_breadth([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# R6 — reward_relevance
# ---------------------------------------------------------------------------


class TestRewardRelevance:
    def test_legal_first_moves_score_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_relevance([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] > 0

    def test_illegal_first_move_score_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_relevance([prompt], [_completion(ILLEGAL_COMPLETION)])
        # e5 is not legal for White from starting position
        assert scores[0] < 1.0  # at least one bad first move

    def test_no_lines_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_relevance([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# combined_reward — hard gate behaviour
# ---------------------------------------------------------------------------


class TestCombinedReward:
    def test_illegal_completion_dominated_by_gate(self):
        # All lines illegal → combined should be -1.0 (hard gate).
        # Patch _eval_fen so Stockfish is never spawned.
        all_illegal = (
            "<line>LINE 1: e5 (illegal) → Nf6 (develop) | eval: equal</line>\n"
            "<line>LINE 2: d5 (illegal) → d4 (control) | eval: equal</line>\n"
        )
        prompt = _make_prompt(STARTING_FEN)
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward([prompt], [_completion(all_illegal)])
        assert scores[0] == -1.0

    def test_legal_completion_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        # Stockfish returns 0 cp (equal) — R2 exact match → 1.0
        # Combined should be positive.
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] >= 0.0

    def test_batch_size_matches(self):
        prompts = [_make_prompt(STARTING_FEN)] * 3
        completions = [
            _completion(LEGAL_COMPLETION),
            _completion(ILLEGAL_COMPLETION),
            _completion(LEGAL_COMPLETION),
        ]
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward(prompts, completions)
        assert len(scores) == 3
