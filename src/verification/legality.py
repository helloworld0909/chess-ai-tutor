"""Move legality validation using python-chess."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import chess


class MoveFormat(Enum):
    """Supported move formats."""

    UCI = "uci"  # e.g., "e2e4", "e7e8q"
    SAN = "san"  # e.g., "e4", "Nf3", "O-O"
    LAN = "lan"  # e.g., "e2-e4", "Ng1-f3"


@dataclass
class MoveValidationResult:
    """Result of move validation."""

    valid: bool
    move_uci: str | None = None
    move_san: str | None = None
    error: str | None = None
    resulting_fen: str | None = None
    is_check: bool = False
    is_checkmate: bool = False
    is_capture: bool = False
    captured_piece: str | None = None


def validate_fen(fen: str) -> tuple[bool, str | None]:
    """Validate a FEN string.

    Args:
        fen: Position in FEN notation

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return False, "Invalid board position (illegal king placement or similar)"
        return True, None
    except ValueError as e:
        return False, f"Invalid FEN format: {e}"


def validate_move(
    fen: str,
    move: str,
    format: MoveFormat = MoveFormat.UCI,
) -> MoveValidationResult:
    """Validate a move and return detailed result.

    Args:
        fen: Position in FEN notation
        move: Move in specified format
        format: Move format (UCI, SAN, or LAN)

    Returns:
        MoveValidationResult with validation details
    """
    # Validate FEN first
    fen_valid, fen_error = validate_fen(fen)
    if not fen_valid:
        return MoveValidationResult(valid=False, error=fen_error)

    board = chess.Board(fen)

    # Parse move based on format
    try:
        if format == MoveFormat.UCI:
            chess_move = chess.Move.from_uci(move)
        elif format == MoveFormat.SAN:
            chess_move = board.parse_san(move)
        elif format == MoveFormat.LAN:
            # Convert LAN to UCI (remove dashes and special characters)
            uci_move = move.replace("-", "").replace("x", "")
            chess_move = chess.Move.from_uci(uci_move)
        else:
            return MoveValidationResult(valid=False, error=f"Unknown format: {format}")
    except ValueError as e:
        return MoveValidationResult(
            valid=False,
            error=f"Cannot parse move '{move}': {e}",
        )

    # Check if move is legal
    if chess_move not in board.legal_moves:
        # Try to give a helpful error message
        if board.is_check():
            return MoveValidationResult(
                valid=False,
                error=f"Move '{move}' is illegal. You are in check and must respond to it.",
            )
        elif board.piece_at(chess_move.from_square) is None:
            return MoveValidationResult(
                valid=False,
                error=f"Move '{move}' is illegal. No piece on the starting square.",
            )
        else:
            return MoveValidationResult(
                valid=False,
                error=f"Move '{move}' is illegal in this position.",
            )

    # Move is valid - gather additional info
    san = board.san(chess_move)
    is_capture = board.is_capture(chess_move)
    captured = board.piece_at(chess_move.to_square) if is_capture else None

    # Apply move to get resulting position
    board.push(chess_move)

    return MoveValidationResult(
        valid=True,
        move_uci=chess_move.uci(),
        move_san=san,
        resulting_fen=board.fen(),
        is_check=board.is_check(),
        is_checkmate=board.is_checkmate(),
        is_capture=is_capture,
        captured_piece=captured.symbol() if captured else None,
    )


def parse_move_flexible(fen: str, move: str) -> MoveValidationResult:
    """Try to parse a move in any format.

    Attempts UCI, SAN, and LAN formats in order.

    Args:
        fen: Position in FEN notation
        move: Move in any supported format

    Returns:
        MoveValidationResult from first successful parse
    """
    # Try each format
    for format in [MoveFormat.UCI, MoveFormat.SAN, MoveFormat.LAN]:
        result = validate_move(fen, move, format)
        if result.valid:
            return result

    # None worked - return the UCI attempt's error
    return validate_move(fen, move, MoveFormat.UCI)


def get_legal_moves_for_piece(fen: str, square: str) -> list[str]:
    """Get all legal moves for a piece on a specific square.

    Args:
        fen: Position in FEN notation
        square: Square name (e.g., "e2")

    Returns:
        List of legal moves in UCI notation
    """
    fen_valid, _ = validate_fen(fen)
    if not fen_valid:
        return []

    board = chess.Board(fen)

    try:
        sq = chess.parse_square(square)
    except ValueError:
        return []

    return [
        move.uci()
        for move in board.legal_moves
        if move.from_square == sq
    ]


def filter_valid_moves(fen: str, moves: Sequence[str]) -> list[str]:
    """Filter a list of moves to only include legal ones.

    Args:
        fen: Position in FEN notation
        moves: List of moves to filter (any format)

    Returns:
        List of valid moves in UCI notation
    """
    valid_moves = []

    for move in moves:
        result = parse_move_flexible(fen, move)
        if result.valid and result.move_uci:
            valid_moves.append(result.move_uci)

    return valid_moves


def is_game_over(fen: str) -> dict:
    """Check if the game is over.

    Args:
        fen: Position in FEN notation

    Returns:
        Dictionary with game state
    """
    fen_valid, fen_error = validate_fen(fen)
    if not fen_valid:
        return {"error": fen_error}

    board = chess.Board(fen)

    return {
        "is_game_over": board.is_game_over(),
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_insufficient_material": board.is_insufficient_material(),
        "is_seventyfive_moves": board.is_seventyfive_moves(),
        "is_fivefold_repetition": board.is_fivefold_repetition(),
        "can_claim_draw": board.can_claim_draw(),
        "outcome": str(board.outcome()) if board.outcome() else None,
    }
