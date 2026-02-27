"""Shared chess coaching tool definitions and implementations.

These tools are available to the LLM at both training time (prepare_datasets.py)
and inference time (web.py), ensuring the model is trained and evaluated on the
same tool interface.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

ANALYZE_POSITION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "analyze_position",
        "description": (
            "Use Stockfish to analyze a chess position. "
            "Returns evaluation, best moves, and principal variation. "
            "Call this to explore what happens after a move or to compare alternatives."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "FEN string of the position to analyze",
                },
                "multipv": {
                    "type": "integer",
                    "description": "Number of top moves to return (1–5, default 3)",
                    "default": 3,
                },
            },
            "required": ["fen"],
        },
    },
}

WEB_SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search for chess-related information online. "
            "Use this to look up opening names and theory, famous game patterns, "
            "endgame techniques, or any chess concept you want to reference. "
            "Returns a list of relevant results with titles and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query using the opening name plus the COMPLETE move "
                        "sequence from move 1 — NOT FEN strings and NOT isolated moves. "
                        "Include every move starting from 1.e4 so the move extractor "
                        "can replay the game. "
                        "Good: 'Ruy Lopez e4 e5 Nf3 Nc6 Bb5 Qe7' "
                        "or 'Italian Game Giuoco Piano e4 e5 Nf3 Nc6 Bc4'. "
                        "Bad: 'Ruy Lopez Bb5 Qe7 theory' (isolated moves, no full sequence). "
                        "Bad: 'rnbqkb1r/pppp1ppp/... opening name' (FEN strings)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

PLAY_MOVES_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_moves",
        "description": (
            "Apply a sequence of moves to a position and return the resulting FEN. "
            "Use this to trace a candidate line step-by-step before committing it. "
            "Returns an error if any move in the sequence is illegal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "Starting position in FEN notation",
                },
                "moves": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Sequence of moves in SAN or UCI notation "
                        "(e.g. ['e4', 'e5', 'Nf3'] or ['e2e4', 'e7e5'])"
                    ),
                },
            },
            "required": ["fen", "moves"],
        },
    },
}

GET_TOP_LINES_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_top_lines",
        "description": (
            "Get Stockfish's top N engine lines from a position. "
            "Each line includes the move sequence in SAN, depth, and evaluation label. "
            "Use this to discover the best continuations before writing your lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "Position to analyse in FEN notation",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of top lines to return (1–5, default 5)",
                    "default": 5,
                },
                "depth": {
                    "type": "integer",
                    "description": "Stockfish search depth (default 15)",
                    "default": 15,
                },
            },
            "required": ["fen"],
        },
    },
}

GET_ATTACKS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_attacks",
        "description": (
            "Return which pieces attack a given square and which squares a piece on "
            "that square attacks. Useful for reasoning about pins, forks, discovered "
            "attacks, and piece safety."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "Position in FEN notation",
                },
                "square": {
                    "type": "string",
                    "description": "Square to inspect in algebraic notation (e.g. 'e4', 'd5')",
                },
            },
            "required": ["fen", "square"],
        },
    },
}

FEN_TO_BOARD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "fen_to_board",
        "description": (
            "Convert a FEN string to a human-readable ASCII board diagram. "
            "Use this to visualise any position during your thinking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "Position in FEN notation",
                },
            },
            "required": ["fen"],
        },
    },
}

BOARD_TO_FEN_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "board_to_fen",
        "description": (
            "Assemble a full FEN string from a piece-placement description. "
            "Provide the piece placement part (rank 8 to rank 1, slash-separated). "
            "Side to move, castling, and en-passant are optional."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "placement": {
                    "type": "string",
                    "description": (
                        "Piece placement string "
                        "(e.g. 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR')"
                    ),
                },
                "side_to_move": {
                    "type": "string",
                    "description": "'w' for White, 'b' for Black (default 'w')",
                    "default": "w",
                },
                "castling": {
                    "type": "string",
                    "description": "Castling availability (e.g. 'KQkq', '-'). Default '-'.",
                    "default": "-",
                },
                "en_passant": {
                    "type": "string",
                    "description": "En-passant target square (e.g. 'e3') or '-'. Default '-'.",
                    "default": "-",
                },
            },
            "required": ["placement"],
        },
    },
}

CHESS_TOOLS: list[dict] = [
    ANALYZE_POSITION_TOOL,
    WEB_SEARCH_TOOL,
    PLAY_MOVES_TOOL,
    GET_TOP_LINES_TOOL,
    GET_ATTACKS_TOOL,
    FEN_TO_BOARD_TOOL,
    BOARD_TO_FEN_TOOL,
]


# ---------------------------------------------------------------------------
# Web search implementation
# ---------------------------------------------------------------------------

# Matches SAN move tokens: e4, d5, Nf3, Bc4, O-O, O-O-O, Bxe5, dxe5=Q, etc.
_SAN_RE = re.compile(
    r"\b(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[KQRBN][a-h][1-8][+#]?)\b"
)


def _extract_play_param(query: str) -> str | None:
    """Parse SAN move tokens from a query string and return a Lichess 'play' parameter.

    Replays the moves on a fresh board so invalid sequences are cut off early.
    Returns comma-separated UCI moves (e.g. 'e2e4,e7e5,g1f3') or None if
    fewer than 2 legal moves were found.
    """
    tokens = _SAN_RE.findall(query)
    board = chess.Board()
    uci_moves: list[str] = []
    for san in tokens:
        try:
            move = board.parse_san(san)
            uci_moves.append(move.uci())
            board.push(move)
        except Exception:
            break  # stop at first illegal token
    return ",".join(uci_moves) if len(uci_moves) >= 2 else None


async def web_search(query: str) -> str:
    """Search for chess information using the Lichess opening explorer and DuckDuckGo.

    Lichess Opening Explorer is queried when the query contains a FEN or a
    recognisable move sequence; it returns authoritative opening names and
    master-game statistics.  DuckDuckGo is used as a general fallback.
    Returns a JSON-encoded list of result dicts with 'title', 'snippet', 'url'.
    """
    import httpx

    results: list[dict] = []

    # 1. Lichess Opening Explorer — authoritative for opening theory
    # Detect an explicit FEN token (position part has 7 slashes, e.g.
    # rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR).
    fen_candidate: str | None = None
    for word in query.split():
        if word.count("/") >= 6 and len(word) > 15:
            fen_candidate = word
            break

    # If no FEN, try to extract a move sequence and build a Lichess play param.
    play_param: str | None = None
    if not fen_candidate:
        play_param = _extract_play_param(query)

    # Only query Lichess when we have a concrete position (FEN or moves).
    # Querying without one returns the starting-position record which has no
    # opening name and is useless.
    if fen_candidate or play_param:
        try:
            params: dict = {"topGames": 3, "recentGames": 0, "moves": 5}
            if fen_candidate:
                params["fen"] = fen_candidate
            else:
                params["play"] = play_param
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get("https://explorer.lichess.ovh/masters", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    opening = data.get("opening") or {}
                    top_moves = data.get("moves", [])[:3]
                    if opening.get("name"):
                        snippet = f"Opening: {opening['name']} (ECO {opening.get('eco', '?')}). "
                        if top_moves:
                            moves_str = ", ".join(
                                f"{m['san']} ({m['white']}W/{m['draws']}D/{m['black']}B)"
                                for m in top_moves
                            )
                            snippet += f"Top master moves: {moves_str}."
                        results.append(
                            {
                                "title": f"Lichess Opening Explorer: {opening['name']}",
                                "snippet": snippet,
                                "url": "https://lichess.org/opening",
                            }
                        )
        except Exception:
            pass

    # 2. DuckDuckGo instant answers
    # Strip FEN tokens (slash-containing words) from the DDG query — they are
    # unreadable by web search and produce zero results.
    ddg_query = " ".join(w for w in query.split() if "/" not in w).strip() or query
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": f"chess {ddg_query}",
                    "format": "json",
                    "no_redirect": 1,
                    "no_html": 1,
                },
                headers={"User-Agent": "chess-ai-tutor/1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("AbstractText"):
                    results.append(
                        {
                            "title": data.get("Heading", "Chess Info"),
                            "snippet": data["AbstractText"][:400],
                            "url": data.get("AbstractURL", ""),
                        }
                    )
                for rel in data.get("RelatedTopics", [])[:3]:
                    if isinstance(rel, dict) and rel.get("Text"):
                        results.append(
                            {
                                "title": rel.get("Text", "")[:80],
                                "snippet": rel.get("Text", "")[:300],
                                "url": rel.get("FirstURL", ""),
                            }
                        )
    except Exception:
        pass

    if not results:
        return json.dumps({"error": "No results found", "query": query})
    return json.dumps(results[:5])


# ---------------------------------------------------------------------------
# play_moves implementation
# ---------------------------------------------------------------------------

# Centipawn → eval label (white perspective) — mirrors generate_lines.py
_CP_BANDS = [
    (300, "winning for white"),
    (101, "good for white"),
    (-100, "equal"),
    (-300, "good for black"),
]


def _cp_to_label(cp: int | None, mate_in: int | None) -> str:
    if mate_in is not None:
        return "winning for white" if mate_in > 0 else "winning for black"
    if cp is None:
        return "equal"
    for threshold, label in _CP_BANDS:
        if cp >= threshold:
            return label
    return "winning for black"


def play_moves(fen: str, moves: list[str]) -> str:
    """Apply a sequence of SAN or UCI moves to a position.

    Returns JSON with the resulting FEN and the move list in SAN.
    Returns an error if any move is illegal or unparseable.
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return json.dumps({"error": f"Invalid FEN: {e}"})

    san_moves: list[str] = []
    for i, mv in enumerate(moves):
        try:
            # Try SAN first, then UCI
            try:
                move = board.parse_san(mv)
            except Exception:
                move = chess.Move.from_uci(mv)
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move: {mv}")
            san_moves.append(board.san(move))
            board.push(move)
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Move {i + 1} ('{mv}') is illegal: {e}",
                    "moves_applied": san_moves,
                    "fen_after_applied": board.fen(),
                }
            )

    return json.dumps(
        {
            "resulting_fen": board.fen(),
            "moves_san": san_moves,
            "is_check": board.is_check(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
            "is_game_over": board.is_game_over(),
        }
    )


# ---------------------------------------------------------------------------
# get_top_lines implementation
# ---------------------------------------------------------------------------


async def get_top_lines(
    fen: str,
    stockfish: "Stockfish",  # type: ignore[name-defined]  # noqa: F821
    n: int = 5,
    depth: int = 15,
) -> str:
    """Return Stockfish's top N lines from a position as JSON.

    Each line contains the move sequence in SAN and an eval label.
    Raw centipawn numbers are never returned — only human-readable labels.
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return json.dumps({"error": f"Invalid FEN: {e}"})

    n = max(1, min(n, 5))
    try:
        analysis = await stockfish.analyze(fen, depth=depth, multipv=n)
    except Exception as e:
        return json.dumps({"error": f"Stockfish error: {e}"})

    lines = []
    for i, line in enumerate(analysis.lines[:n], 1):
        if not line.pv:
            continue
        line_board = board.copy()
        san_moves: list[str] = []
        for uci in line.pv:
            try:
                mv = chess.Move.from_uci(uci)
                san_moves.append(line_board.san(mv))
                line_board.push(mv)
            except Exception:
                break
        label = _cp_to_label(line.score.centipawns, line.score.mate_in)
        lines.append(
            {
                "line": i,
                "moves": san_moves,
                "eval": label,
                "depth": line.depth,
            }
        )

    return json.dumps({"lines": lines, "fen": fen})


# ---------------------------------------------------------------------------
# get_attacks implementation
# ---------------------------------------------------------------------------


def get_attacks(fen: str, square: str) -> str:
    """Return attack relationships for a given square.

    Returns:
      - piece_on_square: piece occupying the square (if any)
      - attacked_by: pieces of both colours that attack this square
      - attacks_to: squares this piece attacks (if a piece is present)
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return json.dumps({"error": f"Invalid FEN: {e}"})

    try:
        sq = chess.parse_square(square)
    except ValueError:
        return json.dumps({"error": f"Invalid square: {square}"})

    piece = board.piece_at(sq)
    piece_str = piece.symbol() if piece else None

    # Which pieces attack this square?
    attacked_by: list[dict] = []
    for color in (chess.WHITE, chess.BLACK):
        for attacker_sq in board.attackers(color, sq):
            ap = board.piece_at(attacker_sq)
            if ap:
                attacked_by.append(
                    {
                        "square": chess.square_name(attacker_sq),
                        "piece": ap.symbol(),
                        "color": "white" if color == chess.WHITE else "black",
                    }
                )

    # Which squares does the piece on this square attack?
    attacks_to: list[str] = []
    if piece is not None:
        attacks_to = [chess.square_name(s) for s in board.attacks(sq)]

    return json.dumps(
        {
            "square": square,
            "piece_on_square": piece_str,
            "attacked_by": attacked_by,
            "attacks_to": attacks_to,
        }
    )


# ---------------------------------------------------------------------------
# fen_to_board / board_to_fen implementations
# ---------------------------------------------------------------------------


def fen_to_board(fen: str) -> str:
    """Return an ASCII board diagram for a FEN position.

    Rank 8 is at the top (Black's side); rank 1 at the bottom (White's side).
    Upper-case = White pieces; lower-case = Black pieces.
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return json.dumps({"error": f"Invalid FEN: {e}"})

    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = f"{rank + 1} "
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            row += (piece.symbol() if piece else ".") + " "
        lines.append(row.rstrip())
    diagram = "\n".join(lines)

    side = "White" if board.turn == chess.WHITE else "Black"
    return json.dumps(
        {
            "board": diagram,
            "fen": fen,
            "side_to_move": side,
        }
    )


def board_to_fen(
    placement: str,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
) -> str:
    """Assemble and validate a FEN string from its components.

    Returns the canonical FEN (normalised by python-chess) or an error.
    """
    side = side_to_move.strip().lower()
    if side not in ("w", "b"):
        return json.dumps({"error": f"Invalid side_to_move: '{side_to_move}' — use 'w' or 'b'"})
    fen = f"{placement} {side} {castling} {en_passant} 0 1"
    try:
        board = chess.Board(fen)
    except ValueError as e:
        return json.dumps({"error": f"Invalid FEN components: {e}", "attempted_fen": fen})
    return json.dumps({"fen": board.fen()})
