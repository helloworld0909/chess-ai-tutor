"""FastAPI web server for Chess Game Review UI.

Serves the frontend and provides API endpoints for game navigation
and move analysis using Stockfish.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_mcp.representations import render_board_svg
from chess_mcp.stockfish import Stockfish

# Global state – populated by review.py before server starts
_games: list[Any] = []
_stockfish: Stockfish | None = None
_username: str = ""

STATIC_DIR = Path(__file__).parent.parent.parent / "static"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _stockfish
    stockfish_path = os.environ.get("STOCKFISH_PATH")
    _stockfish = Stockfish(path=stockfish_path)
    await _stockfish.start()
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


class AnalyzeResponse(BaseModel):
    classification: str
    best_move: str
    cp_loss: int
    is_best: bool
    eval_before: str  # e.g. "+0.35" or "M3"
    comment: str


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


def _format_score(value: int, is_mate: bool) -> str:
    if is_mate:
        return f"M{abs(value)}" if value > 0 else f"-M{abs(value)}"
    return f"{value / 100:+.2f}"


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

    # Get the eval before the move
    analysis = await _stockfish.analyze(req.fen, depth=16, multipv=1)
    score = analysis.score
    eval_str = _format_score(
        score.mate_in if score.mate_in is not None else (score.centipawns or 0),
        score.mate_in is not None,
    )

    # Get SAN for the comment
    import chess

    board = chess.Board(req.fen)
    try:
        san = board.san(chess.Move.from_uci(req.move_uci))
    except Exception:
        san = req.move_uci

    comment = _generate_comment(san, classification, best_move, is_best)

    return AnalyzeResponse(
        classification=classification,
        best_move=best_move,
        cp_loss=cp_loss,
        is_best=is_best,
        eval_before=eval_str,
        comment=comment,
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
