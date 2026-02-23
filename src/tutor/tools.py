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

CHESS_TOOLS: list[dict] = [ANALYZE_POSITION_TOOL, WEB_SEARCH_TOOL]


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
