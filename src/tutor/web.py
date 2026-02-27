"""FastAPI web server for Chess Game Review UI.

Serves the frontend and provides API endpoints for game navigation
and move analysis using Stockfish.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from chess_mcp.representations import render_board_svg
from chess_mcp.stockfish import Stockfish
from tutor.analysis import build_move_tree, compute_cct, compute_position_context, format_score
from tutor.prompts import (
    SYSTEM_PROMPT,
    board_ascii,
    format_user_prompt,
    move_facts,
)
from tutor.tools import CHESS_TOOLS
from tutor.tools import web_search as _web_search

# Global state – populated by review.py before server starts
_games: list[Any] = []
_stockfish: Stockfish | None = None
_username: str = ""
_llm_client: AsyncOpenAI | None = None
_llm_model: str = ""

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _stockfish, _llm_client, _llm_model
    stockfish_path = os.environ.get("STOCKFISH_PATH")
    _stockfish = Stockfish(path=stockfish_path)
    await _stockfish.start()

    llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8100/v1")
    _llm_model = os.environ.get("LLM_MODEL", "chess-tutor")
    _llm_client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")

    yield
    try:
        await _stockfish.stop()
    except Exception:
        pass  # Ignore errors on shutdown (e.g. pipe already closed)


app = FastAPI(title="Chess Game Review", lifespan=lifespan)

# Serve static files (JS, CSS) at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MoveOut(BaseModel):
    index: int
    san: str
    uci: str
    fen_before: str
    color: str
    move_number: int


class GameSummary(BaseModel):
    id: str
    url: str
    white: str
    black: str
    result: str
    result_detail: str
    date: str
    time_control: str
    title: str
    move_count: int


class GameDetail(BaseModel):
    id: str
    url: str
    white: str
    black: str
    result: str
    result_detail: str
    date: str
    time_control: str
    moves: list[MoveOut]
    final_fen: str


class AnalyzeRequest(BaseModel):
    fen: str
    move_uci: str
    model: str | None = None  # None = use server default (_llm_model)


class AnalyzeResponse(BaseModel):
    classification: str
    best_move: str
    cp_loss: int
    is_best: bool
    eval_before: str  # e.g. "+0.35" or "M3"
    comment: str
    comment_source: str  # "llm" or "template"
    thinking: str  # raw <think> block content, empty string if none


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _generate_comment(
    san: str,
    classification: str,
    best_move: str,
    is_best: bool,
) -> str:
    """Generate a short human-readable comment for a move."""
    if is_best:
        return (
            f"{san} is the engine's top choice. "
            "This move optimally addresses the demands of the position."
        )
    if classification in ("Best", "Great", "Good"):
        return (
            f"{san} is a reasonable move. "
            f"The engine slightly prefers {best_move}, but yours is perfectly playable."
        )
    if classification == "Inaccuracy":
        return (
            f"{san} is slightly imprecise. "
            f"Consider {best_move}, which better addresses the position's needs."
        )
    if classification == "Mistake":
        return (
            f"{san} is a mistake that gives your opponent an advantage. "
            f"The correct move was {best_move}."
        )
    if classification == "Blunder":
        return (
            f"{san} is a significant blunder! "
            f"You should have played {best_move} instead. "
            "Always check for tactics before committing to a move."
        )
    return f"Move played: {san}."


async def _llm_comment(
    san: str,
    classification: str,
    best_move: str,
    cp_loss: int,
    eval_str: str,
    candidates: list[str],
    opponent_threats: list[str],
    board_ascii_str: str = "",
    fen: str = "",
    cct: dict[str, list[str]] | None = None,
    move_tree: list[Any] | None = None,
    position_context: dict[str, Any] | None = None,
    facts: list[str] | None = None,
    model: str | None = None,
) -> tuple[str, str, str]:  # (comment, source, thinking)
    """Get an AI-generated comment for a move using a tool-calling loop.

    The LLM can call analyze_position (via Stockfish) and web_search to ground
    its comment before writing. Returns (comment text, source) where source is
    "llm" or "template".
    """
    if _llm_client is None:
        return _generate_comment(san, classification, best_move, cp_loss == 0), "template", ""

    import json as _json

    import chess as _chess

    effective_model = model or _llm_model

    prompt = format_user_prompt(
        board_ascii_str=board_ascii_str,
        san=san,
        classification=classification,
        eval_str=eval_str,
        best_move=best_move,
        cp_loss=cp_loss,
        candidates=candidates,
        opponent_threats=opponent_threats,
        facts=facts,
        fen=fen,
        cct=cct,
        move_tree=move_tree,
        position_context=position_context,
    )

    logger.debug(
        "LLM context for move %s (%s):\n--- SYSTEM ---\n%s\n--- USER ---\n%s",
        san,
        classification,
        SYSTEM_PROMPT,
        prompt,
    )

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        for _ in range(6):  # up to 4 tool calls + 1 answer + 1 safety
            response = await asyncio.wait_for(
                _llm_client.chat.completions.create(  # type: ignore[call-overload]
                    model=effective_model,
                    messages=messages,
                    tools=CHESS_TOOLS,
                    tool_choice="auto",
                    max_tokens=8192,
                    temperature=0.7,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                ),
                timeout=300.0,
            )
            choice = response.choices[0]
            msg = choice.message

            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                logger.debug(
                    "LLM tool call (move %s): %s",
                    san,
                    ", ".join(
                        f"{tc.function.name}({tc.function.arguments})" for tc in msg.tool_calls
                    ),
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    try:
                        args = _json.loads(tc.function.arguments)
                        if tc.function.name == "web_search":
                            result_str = await _web_search(args.get("query", ""))
                        else:  # analyze_position
                            if _stockfish is None:
                                result_str = _json.dumps({"error": "Stockfish not ready"})
                            else:
                                board_q = _chess.Board(args.get("fen", fen))
                                analysis = await _stockfish.analyze(
                                    board_q.fen(),
                                    depth=16,
                                    multipv=min(int(args.get("multipv", 3)), 5),
                                )
                                result_str = _json.dumps(
                                    {
                                        "eval": format_score(
                                            analysis.score.mate_in
                                            if analysis.score.mate_in is not None
                                            else (analysis.score.centipawns or 0),
                                            analysis.score.mate_in is not None,
                                        ),
                                        "lines": [
                                            {"move": ln.best_move, "pv": ln.pv[:5]}
                                            for ln in analysis.lines[:5]
                                        ],
                                    }
                                )
                    except Exception as e:
                        result_str = _json.dumps({"error": str(e)})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                continue

            # finish_reason == "stop" — extract <think> block, strip it from comment
            raw = msg.content or ""
            logger.debug(
                "LLM raw output (move %s):\n--- RESPONSE ---\n%s",
                san,
                raw,
            )
            think_match = _THINK_RE.search(raw)
            thinking = think_match.group(1).strip() if think_match else ""
            text = _THINK_RE.sub("", raw).strip()
            if text:
                return text, "llm", thinking
            break
    except Exception:
        pass

    return _generate_comment(san, classification, best_move, cp_loss == 0), "template", ""


# format_score imported from tutor.analysis


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    """Serve the main page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/games", response_model=list[GameSummary])
async def list_games() -> list[GameSummary]:
    """Return summary of all loaded games."""
    from tutor.chesscom import format_game_title

    result = []
    for g in _games:
        result.append(
            GameSummary(
                id=g.id,
                url=g.url,
                white=g.white,
                black=g.black,
                result=g.result,
                result_detail=g.result_detail,
                date=g.date,
                time_control=g.time_control,
                title=format_game_title(g, _username),
                move_count=len(g.moves),
            )
        )
    return result


@app.get("/api/game/{game_id}", response_model=GameDetail)
async def get_game(game_id: str) -> GameDetail:
    """Return full move list and FEN positions for a game."""
    game = next((g for g in _games if g.id == game_id), None)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    return GameDetail(
        id=game.id,
        url=game.url,
        white=game.white,
        black=game.black,
        result=game.result,
        result_detail=game.result_detail,
        date=game.date,
        time_control=game.time_control,
        moves=[MoveOut(**m.__dict__) for m in game.moves],
        final_fen=game.final_fen,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_move(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a move using Stockfish.

    Args:
        req: FEN before the move and the move in UCI notation

    Returns:
        Classification, best move, centipawn loss, and a comment
    """
    if _stockfish is None:
        raise HTTPException(status_code=503, detail="Stockfish not ready")

    comparison = await _stockfish.compare_moves(req.fen, req.move_uci)

    if "error" in comparison:
        raise HTTPException(status_code=400, detail=comparison["error"])

    classification: str = comparison["classification"]
    best_move: str = comparison["best_move"]
    cp_loss: int = comparison["cp_loss"]
    is_best: bool = comparison["is_best"]

    # Get the eval + top candidates before the move, and opponent threats — run in parallel
    import chess

    analysis, threats_data = await asyncio.gather(
        _stockfish.analyze(req.fen, depth=16, multipv=3),
        _stockfish.get_threats(req.fen),
    )
    score = analysis.score
    eval_str = format_score(
        score.mate_in if score.mate_in is not None else (score.centipawns or 0),
        score.mate_in is not None,
    )

    # Convert top candidate moves to SAN for the LLM
    board = chess.Board(req.fen)
    candidates: list[str] = []
    for line in analysis.lines[:3]:
        try:
            move_san = board.san(chess.Move.from_uci(line.best_move))
            cp = line.score.centipawns
            score_str = format_score(
                line.score.mate_in if line.score.mate_in is not None else (cp or 0),
                line.score.mate_in is not None,
            )
            candidates.append(f"{move_san} ({score_str})")
        except Exception:
            pass

    # Convert opponent threats to SAN (threats use a null-move board, so flip turn)
    opponent_threats: list[str] = []
    try:
        threat_board = chess.Board(req.fen)
        threat_board.push(chess.Move.null())
        for t in threats_data.get("threats", [])[:2]:
            try:
                threat_san = threat_board.san(chess.Move.from_uci(t["move"]))
                opponent_threats.append(threat_san)
            except Exception:
                pass
    except Exception:
        pass

    # SAN for the played move
    try:
        san = board.san(chess.Move.from_uci(req.move_uci))
    except Exception:
        san = req.move_uci

    # Compute CCT and positional context for the position before the move
    try:
        cct = compute_cct(board)
    except Exception:
        cct = {}

    try:
        pos_ctx = compute_position_context(board)
    except Exception:
        pos_ctx = {}

    # Compute deterministic move facts (captures, checks, castling, etc.)
    try:
        facts = move_facts(board, chess.Move.from_uci(req.move_uci))
    except Exception:
        facts = []

    # Build 3-level continuation tree (depth=12 to keep inference latency reasonable)
    board_after = board.copy()
    board_after.push(chess.Move.from_uci(req.move_uci))
    try:
        move_tree = await build_move_tree(_stockfish, board_after, depth=12)
    except Exception:
        move_tree = []

    comment, comment_source, thinking = await _llm_comment(
        san,
        classification,
        best_move,
        cp_loss,
        eval_str,
        candidates,
        opponent_threats,
        board_ascii_str=board_ascii(board),
        fen=req.fen,
        cct=cct,
        move_tree=move_tree,
        position_context=pos_ctx,
        facts=facts,
        model=req.model,
    )

    return AnalyzeResponse(
        classification=classification,
        best_move=best_move,
        cp_loss=cp_loss,
        is_best=is_best,
        eval_before=eval_str,
        comment=comment,
        comment_source=comment_source,
        thinking=thinking,
    )


@app.get("/api/username")
async def get_username() -> dict[str, str]:
    """Return the username whose games are loaded."""
    return {"username": _username}


@app.get("/api/board")
async def get_board_svg(
    fen: str = Query(..., description="FEN string of the position"),
    last_move: str | None = Query(None, description="Last move in UCI notation"),
    flip: bool = Query(False, description="Show from Black's perspective"),
    size: int = Query(400, description="Board size in pixels"),
) -> Response:
    """Return an SVG image of the board position."""
    try:
        svg = render_board_svg(fen, size=size, last_move=last_move, flip=flip)
        return Response(content=svg, media_type="image/svg+xml")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
