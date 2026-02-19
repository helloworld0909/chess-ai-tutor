"""Chess.com API client and PGN parser.

Fetches recent games for a player and parses them into structured data.
"""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import chess
import chess.pgn
import httpx

BASE_URL = "https://api.chess.com/pub/player"

# chess.com requires a User-Agent header or requests get rejected
HEADERS = {"User-Agent": "chess-ai-tutor/0.1 (github.com/chess-ai-tutor)"}


@dataclass
class Move:
    """A single move in a game."""

    index: int  # 0-based (0 = white's first move)
    san: str  # Standard algebraic notation, e.g. "e4"
    uci: str  # UCI notation, e.g. "e2e4"
    fen_before: str  # FEN of the position before this move
    color: str  # "white" or "black"
    move_number: int  # 1-based full move number


@dataclass
class Game:
    """A parsed chess.com game."""

    id: str
    url: str
    white: str
    black: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    result_detail: str  # e.g. "resignation", "checkmate", "timeout"
    date: str  # "2024.01.15"
    time_control: str
    moves: list[Move] = field(default_factory=list)
    final_fen: str = chess.STARTING_FEN


def _parse_result_detail(headers: dict[str, str]) -> str:
    """Extract result detail from PGN headers."""
    termination = headers.get("Termination", "")
    if "checkmate" in termination.lower():
        return "checkmate"
    if "resignation" in termination.lower():
        return "resignation"
    if "timeout" in termination.lower() or "time" in termination.lower():
        return "timeout"
    if "agreement" in termination.lower():
        return "agreement"
    if "stalemate" in termination.lower():
        return "stalemate"
    return termination or "unknown"


def parse_pgn(pgn_str: str, game_url: str = "") -> Game | None:
    """Parse a PGN string into a Game object.

    Args:
        pgn_str: Raw PGN string
        game_url: Original chess.com game URL

    Returns:
        Parsed Game, or None if parsing fails
    """
    try:
        pgn_io = io.StringIO(pgn_str)
        pgn_game = chess.pgn.read_game(pgn_io)
        if pgn_game is None:
            return None

        headers = dict(pgn_game.headers)
        board = pgn_game.board()

        moves: list[Move] = []
        for i, node in enumerate(pgn_game.mainline()):
            move = node.move
            if move is None:
                continue
            san = board.san(move)
            uci = move.uci()
            fen_before = board.fen()
            color = "white" if board.turn == chess.WHITE else "black"
            move_number = board.fullmove_number
            moves.append(
                Move(
                    index=i,
                    san=san,
                    uci=uci,
                    fen_before=fen_before,
                    color=color,
                    move_number=move_number,
                )
            )
            board.push(move)

        # Extract date from UTCDate or Date header
        date_str = headers.get("UTCDate", headers.get("Date", "?"))

        return Game(
            id=str(uuid.uuid4()),
            url=game_url or headers.get("Link", ""),
            white=headers.get("White", "?"),
            black=headers.get("Black", "?"),
            result=headers.get("Result", "?"),
            result_detail=_parse_result_detail(headers),
            date=date_str,
            time_control=headers.get("TimeControl", "?"),
            moves=moves,
            final_fen=board.fen(),
        )
    except Exception:
        return None


async def fetch_recent_games(username: str, months: int = 1) -> list[Game]:
    """Fetch recent games for a chess.com username.

    Args:
        username: chess.com username (case-insensitive)
        months: Number of recent months to fetch (1-12)

    Returns:
        List of parsed games, newest first

    Raises:
        httpx.HTTPStatusError: If user not found (404) or API error
        httpx.RequestError: If network error
    """
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0, follow_redirects=True) as client:
        # Get list of monthly archive URLs
        archives_url = f"{BASE_URL}/{username}/games/archives"
        resp = await client.get(archives_url)
        resp.raise_for_status()

        archive_urls: list[str] = resp.json().get("archives", [])
        if not archive_urls:
            return []

        # Take the most recent N months
        recent_archives = archive_urls[-months:]

        games: list[Game] = []
        for archive_url in reversed(recent_archives):  # newest month first
            resp = await client.get(archive_url)
            resp.raise_for_status()

            raw_games = resp.json().get("games", [])
            for raw in reversed(raw_games):  # newest game first
                pgn_str = raw.get("pgn", "")
                if not pgn_str:
                    continue
                game = parse_pgn(pgn_str, raw.get("url", ""))
                if game is not None:
                    games.append(game)

        return games


def format_game_title(game: Game, perspective_user: str) -> str:
    """Format a short title for the game, from the user's perspective.

    Args:
        game: Parsed game
        perspective_user: The username viewing the game

    Returns:
        e.g. "vs. Magnus (White, Won) - 2024.01.15"
    """
    user_lower = perspective_user.lower()
    if game.white.lower() == user_lower:
        color = "White"
        opponent = game.black
        if game.result == "1-0":
            outcome = "Won"
        elif game.result == "0-1":
            outcome = "Lost"
        else:
            outcome = "Draw"
    elif game.black.lower() == user_lower:
        color = "Black"
        opponent = game.white
        if game.result == "0-1":
            outcome = "Won"
        elif game.result == "1-0":
            outcome = "Lost"
        else:
            outcome = "Draw"
    else:
        color = "?"
        opponent = f"{game.white} vs {game.black}"
        outcome = game.result

    return f"vs. {opponent} ({color}, {outcome}) — {game.date}"


def get_current_month_archive_url(username: str) -> str:
    """Return the archive URL for the current month."""
    now = datetime.now()
    return f"{BASE_URL}/{username}/games/{now.year}/{now.month:02d}"
