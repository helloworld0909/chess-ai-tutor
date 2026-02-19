"""Tests for chess.com API client and PGN parser."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tutor.chesscom import (
    Game,
    Move,
    fetch_recent_games,
    format_game_title,
    parse_pgn,
)

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_PGN = """\
[Event "Live Chess"]
[Site "Chess.com"]
[Date "2024.01.15"]
[UTCDate "2024.01.15"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[TimeControl "600"]
[Termination "alice won by resignation"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""

SAMPLE_ARCHIVES = {
    "archives": [
        "https://api.chess.com/pub/player/alice/games/2023/12",
        "https://api.chess.com/pub/player/alice/games/2024/01",
    ]
}

SAMPLE_GAMES_RESPONSE = {
    "games": [
        {
            "url": "https://chess.com/game/123",
            "pgn": SAMPLE_PGN,
        }
    ]
}


# ── PGN parsing ───────────────────────────────────────────────────────────────


def test_parse_pgn_returns_game():
    game = parse_pgn(SAMPLE_PGN, "https://chess.com/game/123")
    assert game is not None
    assert game.white == "alice"
    assert game.black == "bob"
    assert game.result == "1-0"
    assert game.result_detail == "resignation"
    assert game.date == "2024.01.15"
    assert game.time_control == "600"
    assert game.url == "https://chess.com/game/123"


def test_parse_pgn_moves_count():
    game = parse_pgn(SAMPLE_PGN)
    assert game is not None
    # 5 full moves: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 = 10 half-moves
    assert len(game.moves) == 10


def test_parse_pgn_first_move():
    game = parse_pgn(SAMPLE_PGN)
    assert game is not None
    first = game.moves[0]
    assert first.san == "e4"
    assert first.uci == "e2e4"
    assert first.color == "white"
    assert first.move_number == 1
    assert first.index == 0


def test_parse_pgn_move_colors():
    game = parse_pgn(SAMPLE_PGN)
    assert game is not None
    colors = [m.color for m in game.moves]
    # Alternates starting with white
    assert colors[0] == "white"
    assert colors[1] == "black"
    assert colors[2] == "white"


def test_parse_pgn_fen_before_first_move():
    game = parse_pgn(SAMPLE_PGN)
    assert game is not None
    starting_fen_prefix = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"
    assert game.moves[0].fen_before.startswith(starting_fen_prefix)


def test_parse_pgn_returns_none_on_empty():
    result = parse_pgn("")
    assert result is None


def test_parse_pgn_garbage_returns_empty_game():
    # chess.pgn tolerates garbage and returns an empty game with no moves
    result = parse_pgn("this is not a pgn")
    assert result is not None
    assert len(result.moves) == 0


def test_parse_pgn_castling():
    """Verify O-O is parsed correctly."""
    game = parse_pgn(SAMPLE_PGN)
    assert game is not None
    castling_move = next((m for m in game.moves if m.san == "O-O"), None)
    assert castling_move is not None
    assert castling_move.uci == "e1g1"


# ── format_game_title ─────────────────────────────────────────────────────────


def _make_game(white: str, black: str, result: str) -> Game:
    return Game(
        id="test-id",
        url="",
        white=white,
        black=black,
        result=result,
        result_detail="",
        date="2024.01.15",
        time_control="600",
    )


def test_format_game_title_white_win():
    game = _make_game("alice", "bob", "1-0")
    title = format_game_title(game, "alice")
    assert "White" in title
    assert "Won" in title
    assert "bob" in title


def test_format_game_title_black_win():
    game = _make_game("alice", "bob", "0-1")
    title = format_game_title(game, "bob")
    assert "Black" in title
    assert "Won" in title
    assert "alice" in title


def test_format_game_title_loss():
    game = _make_game("alice", "bob", "0-1")
    title = format_game_title(game, "alice")
    assert "Lost" in title


def test_format_game_title_draw():
    game = _make_game("alice", "bob", "1/2-1/2")
    title = format_game_title(game, "alice")
    assert "Draw" in title


def test_format_game_title_case_insensitive():
    game = _make_game("Alice", "Bob", "1-0")
    title = format_game_title(game, "alice")  # lowercase
    assert "Won" in title


# ── fetch_recent_games ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_recent_games_success():
    mock_response_archives = MagicMock()
    mock_response_archives.json.return_value = SAMPLE_ARCHIVES
    mock_response_archives.raise_for_status = MagicMock()

    mock_response_games = MagicMock()
    mock_response_games.json.return_value = SAMPLE_GAMES_RESPONSE
    mock_response_games.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_response_archives, mock_response_games]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("tutor.chesscom.httpx.AsyncClient", return_value=mock_client):
        games = await fetch_recent_games("alice", months=1)

    assert len(games) == 1
    assert games[0].white == "alice"
    assert games[0].black == "bob"


@pytest.mark.asyncio
async def test_fetch_recent_games_no_archives():
    mock_response = MagicMock()
    mock_response.json.return_value = {"archives": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("tutor.chesscom.httpx.AsyncClient", return_value=mock_client):
        games = await fetch_recent_games("nonexistent_user", months=1)

    assert games == []


@pytest.mark.asyncio
async def test_fetch_recent_games_skips_bad_pgn():
    bad_games_response = {
        "games": [
            {"url": "https://chess.com/1", "pgn": ""},  # empty PGN → skipped
            {"url": "https://chess.com/2", "pgn": SAMPLE_PGN},
        ]
    }
    mock_response_archives = MagicMock()
    mock_response_archives.json.return_value = SAMPLE_ARCHIVES
    mock_response_archives.raise_for_status = MagicMock()

    mock_response_games = MagicMock()
    mock_response_games.json.return_value = bad_games_response
    mock_response_games.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_response_archives, mock_response_games]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("tutor.chesscom.httpx.AsyncClient", return_value=mock_client):
        games = await fetch_recent_games("alice", months=1)

    assert len(games) == 1  # Only the valid game
