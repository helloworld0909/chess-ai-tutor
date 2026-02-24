"""Tests for build_textbook_dataset.py."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "pipeline"))

from build_textbook_dataset import (
    _clean_comment,
    _extract_annotators_from_pgn,
    _extract_concepts,
    _filter_users_by_quality,
    _is_quality,
    _parse_pgn_annotations,
)

# ---------------------------------------------------------------------------
# _clean_comment
# ---------------------------------------------------------------------------


def test_clean_comment_strips_lichess_markup():
    raw = "Good move [%cal Ge4e5] because it controls the center."
    assert _clean_comment(raw) == "Good move because it controls the center."


def test_clean_comment_strips_eval_and_clock():
    raw = "Nice [%eval 0.5] [%clk 0:05:00] idea."
    assert _clean_comment(raw) == "Nice idea."


def test_clean_comment_strips_move_number_prefix():
    raw = "1... e5 is the standard reply."
    assert _clean_comment(raw) == "e5 is the standard reply."


def test_clean_comment_normalizes_whitespace():
    raw = "  lots   of   spaces  "
    assert _clean_comment(raw) == "lots of spaces"


# ---------------------------------------------------------------------------
# _is_quality
# ---------------------------------------------------------------------------


def test_is_quality_rejects_short():
    assert not _is_quality("Good move.")


def test_is_quality_accepts_quiz_question():
    # Relaxed filter: quiz/keyword filtering is now the LLM's job
    assert _is_quality("What would you play here? Think about the pawn structure.")


def test_is_quality_accepts_any_prose():
    # No keyword requirement — permissive filter, LLM handles quality
    assert _is_quality(
        "A very interesting position arose from the opening that deserves more study."
    )


def test_is_quality_accepts_instructive():
    text = "This move wins material because it forks the king and rook, threatening both pieces."
    assert _is_quality(text)


def test_is_quality_rejects_game_result():
    assert not _is_quality("1-0")
    assert not _is_quality("0-1")
    assert not _is_quality("1/2-1/2")


def test_is_quality_rejects_pure_move_notation():
    assert not _is_quality("Nxe5+")
    assert not _is_quality("O-O-O")


# ---------------------------------------------------------------------------
# _extract_concepts
# ---------------------------------------------------------------------------


def test_extract_concepts_tactics():
    assert "tactics" in _extract_concepts("This is a classic pin on the long diagonal.")


def test_extract_concepts_endgame():
    assert "endgame" in _extract_concepts(
        "The king and pawn ending is a draw because of the opposition."
    )


def test_extract_concepts_multiple():
    text = "The fork wins material because the passed pawn will promote."
    concepts = _extract_concepts(text)
    assert "tactics" in concepts
    assert "strategy" in concepts


def test_extract_concepts_empty():
    assert _extract_concepts("A random sentence with no chess concepts.") == []


# ---------------------------------------------------------------------------
# _extract_annotators_from_pgn
# ---------------------------------------------------------------------------


def test_extract_annotators_url_form():
    pgn = '[Annotator "https://lichess.org/@/Magnus"]\n1. e4 e5 *'
    assert _extract_annotators_from_pgn(pgn) == {"Magnus"}


def test_extract_annotators_bare_username():
    pgn = '[Annotator "coach_bob"]\n1. e4 e5 *'
    assert _extract_annotators_from_pgn(pgn) == {"coach_bob"}


def test_extract_annotators_rejects_email():
    pgn = '[Annotator "user@example.com"]\n1. e4 e5 *'
    assert _extract_annotators_from_pgn(pgn) == set()


def test_extract_annotators_rejects_long():
    pgn = '[Annotator "this_username_is_way_too_long_for_lichess"]\n1. e4 e5 *'
    assert _extract_annotators_from_pgn(pgn) == set()


def test_extract_annotators_multiple():
    pgn = '[Annotator "https://lichess.org/@/Alice"]\n[Annotator "Bob"]\n1. e4 e5 *'
    assert _extract_annotators_from_pgn(pgn) == {"Alice", "Bob"}


def test_extract_annotators_deduplicates():
    pgn = '[Annotator "Alice"]\n[Annotator "Alice"]\n1. e4 *'
    assert _extract_annotators_from_pgn(pgn) == {"Alice"}


# ---------------------------------------------------------------------------
# _parse_pgn_annotations
# ---------------------------------------------------------------------------

_INSTRUCTIVE_PGN = """\
[Event "Test Study"]
[White "Player A"]
[Black "Player B"]

1. e4 { This move controls the center because it allows piece development. } e5 2. Nf3 { The knight develops naturally, threatening to win the e5 pawn since Black must defend. } *
"""

_NO_ANNOTATION_PGN = """\
[Event "No annotations"]
[White "A"]
[Black "B"]

1. e4 e5 2. Nf3 Nc6 *
"""

_QUIZ_PGN = """\
[Event "Quiz"]
[White "A"]
[Black "B"]

1. e4 { What would you play here? This is a key strategic moment. } e5 *
"""


def test_parse_yields_annotated_positions():
    positions = list(_parse_pgn_annotations(_INSTRUCTIVE_PGN, source="test"))
    assert len(positions) >= 1
    assert all(p.source == "test" for p in positions)
    assert all(p.fen for p in positions)
    assert all(p.move_uci for p in positions)
    assert all(p.annotation for p in positions)


def test_parse_skips_unannotated_moves():
    positions = list(_parse_pgn_annotations(_NO_ANNOTATION_PGN, source="test"))
    assert positions == []


def test_parse_keeps_quiz_annotations():
    # Quiz annotations now pass through — LLM filters them at coaching time
    positions = list(_parse_pgn_annotations(_QUIZ_PGN, source="test"))
    assert len(positions) == 1
    assert "What would you play" in positions[0].annotation


def test_parse_multiple_games():
    two_games = _INSTRUCTIVE_PGN + "\n" + _INSTRUCTIVE_PGN
    positions = list(_parse_pgn_annotations(two_games, source="test"))
    # Should extract from both games (may get duplicates since same game)
    assert len(positions) >= 2


def test_parse_all_annotated_moves_no_cap():
    """With max_per_game removed, all annotated moves in a game are extracted."""
    pgn = "[Event 'T'][White 'A'][Black 'B']\n"
    # Build a game where every move has a long instructive comment
    moves = []
    for i in range(1, 11):
        moves.append(
            f"1. e4 {{ Move {i}: this is instructive because it controls the center allowing development. }}"
        )
    # Use a realistic game with 10+ annotated moves
    long_pgn = """\
[Event "Long"]
[White "A"]
[Black "B"]

1. e4 { First move controls the center because it opens diagonals. }
1... e5 { Black mirrors because symmetry is solid. }
2. Nf3 { Develops the knight because it attacks e5 naturally. }
2... Nc6 { Defends e5 because the knight is active. }
3. Bb5 { The Ruy Lopez because it pins the defender of e5. }
3... a6 { Forces the bishop back because the pin is annoying. }
4. Ba4 { Maintains the pin because retreat keeps pressure. }
4... Nf6 { Attacks e4 because Black wants counterplay. }
5. O-O { Castles because king safety is paramount. }
5... Be7 { Prepares castling because development is required. }
*
"""
    positions = list(_parse_pgn_annotations(long_pgn, source="test"))
    # Should get all 10 annotated moves (no artificial 8-cap)
    assert len(positions) == 10


def test_parse_bad_pgn_doesnt_crash():
    positions = list(_parse_pgn_annotations("this is not pgn at all !!!", source="test"))
    assert positions == []


def test_parse_sets_correct_fen_and_uci():
    positions = list(_parse_pgn_annotations(_INSTRUCTIVE_PGN, source="test"))
    assert len(positions) >= 1
    # First annotated move is 1.e4 — FEN should be starting position
    first = positions[0]
    assert first.move_san == "e4"
    assert first.move_uci == "e2e4"
    assert "rnbqkbnr" in first.fen  # starting position


# ---------------------------------------------------------------------------
# _filter_users_by_quality (async, mocked HTTP)
# ---------------------------------------------------------------------------


def _make_mock_client(response_json: list) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_json
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


@pytest.mark.asyncio
async def test_filter_accepts_titled_player():
    client = _make_mock_client(
        [
            {"username": "GrandMaster1", "title": "GM", "followersCount": 5, "perfs": {}},
        ]
    )
    result = await _filter_users_by_quality(client, ["GrandMaster1"])
    assert "GrandMaster1" in result


@pytest.mark.asyncio
async def test_filter_accepts_high_rating():
    client = _make_mock_client(
        [
            {
                "username": "StrongPlayer",
                "title": "",
                "followersCount": 10,
                "perfs": {"rapid": {"rating": 2200}, "blitz": {"rating": 2100}},
            },
        ]
    )
    result = await _filter_users_by_quality(client, ["StrongPlayer"])
    assert "StrongPlayer" in result


@pytest.mark.asyncio
async def test_filter_accepts_popular_streamer():
    client = _make_mock_client(
        [
            {
                "username": "PopularStreamer",
                "title": "",
                "followersCount": 500,
                "perfs": {"rapid": {"rating": 1500}},
            },
        ]
    )
    result = await _filter_users_by_quality(client, ["PopularStreamer"])
    assert "PopularStreamer" in result


@pytest.mark.asyncio
async def test_filter_rejects_weak_player():
    client = _make_mock_client(
        [
            {
                "username": "WeakPlayer",
                "title": "",
                "followersCount": 3,
                "perfs": {"rapid": {"rating": 1200}, "blitz": {"rating": 1100}},
            },
        ]
    )
    result = await _filter_users_by_quality(client, ["WeakPlayer"])
    assert "WeakPlayer" not in result


@pytest.mark.asyncio
async def test_filter_empty_input():
    client = _make_mock_client([])
    result = await _filter_users_by_quality(client, [])
    assert result == []


@pytest.mark.asyncio
async def test_filter_falls_back_on_api_error():
    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=Exception("network error"))
    result = await _filter_users_by_quality(mock_client, ["user1", "user2"])
    # Permissive fallback — accept all when API fails
    assert set(result) == {"user1", "user2"}
