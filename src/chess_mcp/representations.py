"""Multi-representation converters for chess positions.

Provides FEN, ASCII board, piece-square list, and image rendering
to give the LLM spatial awareness of chess positions.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chess
import chess.svg


@dataclass
class PieceSquare:
    """A piece on a square."""

    piece: str  # e.g., "K", "q", "P"
    square: str  # e.g., "e1", "d8"

    @property
    def color(self) -> Literal["white", "black"]:
        return "white" if self.piece.isupper() else "black"

    @property
    def piece_name(self) -> str:
        names = {
            "k": "king",
            "q": "queen",
            "r": "rook",
            "b": "bishop",
            "n": "knight",
            "p": "pawn",
        }
        return names.get(self.piece.lower(), "unknown")


@dataclass
class PositionRepresentations:
    """All representations of a chess position."""

    fen: str
    ascii_board: str
    piece_squares: dict[str, list[dict[str, str]]]
    side_to_move: str
    castling_rights: str
    en_passant: str | None
    halfmove_clock: int
    fullmove_number: int


def fen_to_ascii(fen: str, flip: bool = False) -> str:
    """Convert FEN to ASCII board representation.

    Args:
        fen: Position in FEN notation
        flip: If True, show board from Black's perspective

    Returns:
        ASCII representation of the board

    Example output:
        8 | r n b q k b n r
        7 | p p p p p p p p
        6 | . . . . . . . .
        5 | . . . . . . . .
        4 | . . . . P . . .
        3 | . . . . . . . .
        2 | P P P P . P P P
        1 | R N B Q K B N R
          +-----------------
            a b c d e f g h
    """
    board = chess.Board(fen)

    lines = []
    ranks = range(7, -1, -1) if not flip else range(8)

    for rank in ranks:
        row = []
        files = range(8) if not flip else range(7, -1, -1)
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                row.append(piece.symbol())
            else:
                row.append(".")
        rank_label = rank + 1
        lines.append(f"{rank_label} | {' '.join(row)}")

    lines.append("  +-----------------")
    if not flip:
        lines.append("    a b c d e f g h")
    else:
        lines.append("    h g f e d c b a")

    return "\n".join(lines)


def fen_to_piece_squares(fen: str) -> dict[str, list[dict[str, str]]]:
    """Convert FEN to piece-square list representation.

    Args:
        fen: Position in FEN notation

    Returns:
        Dictionary with 'white' and 'black' lists of pieces

    Example output:
        {
            "white": [
                {"piece": "K", "square": "e1", "name": "king"},
                {"piece": "Q", "square": "d1", "name": "queen"},
                ...
            ],
            "black": [
                {"piece": "k", "square": "e8", "name": "king"},
                ...
            ]
        }
    """
    board = chess.Board(fen)

    white_pieces = []
    black_pieces = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            piece_info = {
                "piece": piece.symbol(),
                "square": square_name,
                "name": chess.piece_name(piece.piece_type),
            }
            if piece.color == chess.WHITE:
                white_pieces.append(piece_info)
            else:
                black_pieces.append(piece_info)

    # Sort by piece value (K, Q, R, B, N, P)
    piece_order = {"k": 0, "q": 1, "r": 2, "b": 3, "n": 4, "p": 5}

    def sort_key(p):
        return piece_order.get(p["piece"].lower(), 6)

    white_pieces.sort(key=sort_key)
    black_pieces.sort(key=sort_key)

    return {"white": white_pieces, "black": black_pieces}


def get_all_representations(fen: str) -> PositionRepresentations:
    """Get all representations of a position.

    Args:
        fen: Position in FEN notation

    Returns:
        PositionRepresentations with all formats
    """
    board = chess.Board(fen)
    parts = fen.split()

    return PositionRepresentations(
        fen=fen,
        ascii_board=fen_to_ascii(fen),
        piece_squares=fen_to_piece_squares(fen),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        castling_rights=parts[2] if len(parts) > 2 else "-",
        en_passant=parts[3] if len(parts) > 3 and parts[3] != "-" else None,
        halfmove_clock=int(parts[4]) if len(parts) > 4 else 0,
        fullmove_number=int(parts[5]) if len(parts) > 5 else 1,
    )


def render_board_svg(
    fen: str,
    size: int = 400,
    last_move: str | None = None,
    arrows: list[tuple[str, str]] | None = None,
    flip: bool = False,
) -> str:
    """Render board as SVG.

    Args:
        fen: Position in FEN notation
        size: Board size in pixels
        last_move: Last move in UCI notation to highlight
        arrows: List of (from_square, to_square) for arrows
        flip: If True, show from Black's perspective

    Returns:
        SVG string
    """
    board = chess.Board(fen)

    # Convert last move to chess.Move
    last_move_obj = None
    if last_move:
        try:
            last_move_obj = chess.Move.from_uci(last_move)
        except ValueError:
            pass

    # Convert arrows
    arrow_list = []
    if arrows:
        for from_sq, to_sq in arrows:
            try:
                arrow_list.append(
                    chess.svg.Arrow(
                        chess.parse_square(from_sq),
                        chess.parse_square(to_sq),
                    )
                )
            except ValueError:
                pass

    return chess.svg.board(
        board,
        size=size,
        lastmove=last_move_obj,
        arrows=arrow_list,
        flipped=flip,
    )


def render_board_png(
    fen: str,
    output_path: str | Path | None = None,
    size: int = 400,
    last_move: str | None = None,
    arrows: list[tuple[str, str]] | None = None,
    flip: bool = False,
) -> bytes:
    """Render board as PNG image.

    Args:
        fen: Position in FEN notation
        output_path: Optional path to save PNG
        size: Board size in pixels
        last_move: Last move in UCI notation to highlight
        arrows: List of (from_square, to_square) for arrows
        flip: If True, show from Black's perspective

    Returns:
        PNG bytes

    Requires:
        cairosvg package for SVG to PNG conversion
    """
    try:
        import cairosvg
    except ImportError:
        raise ImportError("cairosvg is required for PNG rendering. Install with: pip install cairosvg")

    svg = render_board_svg(fen, size, last_move, arrows, flip)
    png_bytes = cairosvg.svg2png(bytestring=svg.encode())

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_bytes)

    return png_bytes


def format_for_llm(fen: str, include_image_placeholder: bool = False) -> str:
    """Format position for LLM input.

    Creates a comprehensive text representation that helps the LLM
    understand the spatial layout of the position.

    Args:
        fen: Position in FEN notation
        include_image_placeholder: Include [IMAGE] placeholder for VL models

    Returns:
        Formatted string for LLM consumption
    """
    reps = get_all_representations(fen)

    lines = []

    if include_image_placeholder:
        lines.append("[IMAGE: Chess board diagram]")
        lines.append("")

    lines.append(f"**Position** (FEN): `{fen}`")
    lines.append(f"**Side to move**: {reps.side_to_move.capitalize()}")

    if reps.castling_rights != "-":
        castling_desc = []
        if "K" in reps.castling_rights:
            castling_desc.append("White kingside")
        if "Q" in reps.castling_rights:
            castling_desc.append("White queenside")
        if "k" in reps.castling_rights:
            castling_desc.append("Black kingside")
        if "q" in reps.castling_rights:
            castling_desc.append("Black queenside")
        lines.append(f"**Castling available**: {', '.join(castling_desc)}")

    if reps.en_passant:
        lines.append(f"**En passant**: {reps.en_passant}")

    lines.append("")
    lines.append("**Board:**")
    lines.append("```")
    lines.append(reps.ascii_board)
    lines.append("```")

    lines.append("")
    lines.append("**Pieces:**")

    white_pieces = reps.piece_squares["white"]
    black_pieces = reps.piece_squares["black"]

    white_desc = ", ".join(f"{p['name'].capitalize()} on {p['square']}" for p in white_pieces)
    black_desc = ", ".join(f"{p['name'].capitalize()} on {p['square']}" for p in black_pieces)

    lines.append(f"- White: {white_desc}")
    lines.append(f"- Black: {black_desc}")

    return "\n".join(lines)


def create_training_context(
    fen: str,
    user_move: str | None = None,
    question: str | None = None,
) -> dict:
    """Create a training context dictionary for dataset generation.

    Args:
        fen: Position in FEN notation
        user_move: Optional move played by user
        question: Optional question about the position

    Returns:
        Dictionary suitable for training data generation
    """
    reps = get_all_representations(fen)

    context = {
        "fen": fen,
        "ascii_board": reps.ascii_board,
        "piece_squares": reps.piece_squares,
        "side_to_move": reps.side_to_move,
        "castling_rights": reps.castling_rights,
        "en_passant": reps.en_passant,
    }

    if user_move:
        context["user_move"] = user_move

        # Add move details
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                context["user_move_san"] = board.san(move)
                context["is_capture"] = board.is_capture(move)
                context["is_check"] = board.gives_check(move)
        except ValueError:
            pass

    if question:
        context["question"] = question

    return context
