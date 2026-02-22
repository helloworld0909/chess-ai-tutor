"""Shared prompt constants and utilities for chess coaching.

Used by both the inference server (web.py) and the training data pipeline
(prepare_datasets.py) to guarantee training ↔ inference format alignment.
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# System prompt — used verbatim at both training and inference time
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert chess coach giving move-by-move feedback.\n"
    "Rules:\n"
    "- Your comment MUST agree with the engine classification. "
    "If the move is 'Best' or 'Great', explain why it is strong. "
    "If it is 'Good', note it is solid. "
    "If it is 'Inaccuracy', explain the missed opportunity. "
    "If it is 'Mistake' or 'Blunder', explain the error clearly.\n"
    "- Never say a 'Best' move is suboptimal or could be improved.\n"
    "- Do not repeat the classification label or the eval number.\n"
    "- Explain the deeper chess idea: opening theory, strategic plans, "
    "tactical motifs, pawn structure, piece activity, king safety.\n"
    "- The verified move facts are ground truth — do not contradict them, "
    "but go beyond them to explain *why* the move is good or bad.\n"
    "- 2-3 sentences maximum."
)


# ---------------------------------------------------------------------------
# Board / move utilities
# ---------------------------------------------------------------------------


def board_ascii(board: chess.Board) -> str:
    """Return a labelled ASCII board (rank/file labels) for LLM context."""
    rows = str(board).split("\n")  # rank 8 at top
    lines = ["  a b c d e f g h"]
    for i, row in enumerate(rows):
        lines.append(f"{8 - i} {row}")
    lines.append(f"  ({'White' if board.turn == chess.WHITE else 'Black'} to move)")
    return "\n".join(lines)


def move_facts(board: chess.Board, move: chess.Move) -> list[str]:
    """Extract verifiable mechanical facts about a move using python-chess."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return []

    facts: list[str] = []
    piece_name = chess.piece_name(piece.piece_type)
    our_color = piece.color
    their_color = not our_color
    to_sq = chess.square_name(move.to_square)

    # Special moves
    if board.is_castling(move):
        facts.append("king castles for safety and connects the rooks")
        return facts
    if board.is_en_passant(move):
        facts.append(f"en passant capture on {to_sq}")
    elif board.is_capture(move):
        cap = board.piece_at(move.to_square)
        if cap:
            facts.append(f"captures {chess.piece_name(cap.piece_type)} on {to_sq}")

    # Check
    if board.gives_check(move):
        facts.append("gives check to the opponent's king")

    # Was the moving piece under attack before? (defensive retreat)
    if board.attackers(their_color, move.from_square):
        facts.append(f"rescues the {piece_name} which was under attack")

    # Analyse the resulting position
    board_after = board.copy()
    board_after.push(move)

    attacked_sq = board_after.attacks(move.to_square)

    # Enemy pieces now attacked
    attacked_enemies = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in attacked_sq
        if (p := board_after.piece_at(sq)) and p.color == their_color
    ]
    if attacked_enemies:
        facts.append(f"now attacks {', '.join(attacked_enemies)}")

    # Own pieces now supported by the moved piece
    supported = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in attacked_sq
        if (p := board_after.piece_at(sq)) and p.color == our_color and sq != move.to_square
    ]
    if supported:
        facts.append(f"defends own {', '.join(supported)}")

    # Own pieces that lost their defender after this piece moved away
    now_hanging = [
        f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}"
        for sq in board.attacks(move.from_square)
        if (p := board.piece_at(sq))
        and p.color == our_color
        and sq != move.to_square
        and not board_after.is_attacked_by(our_color, sq)  # no longer defended
        and board_after.is_attacked_by(their_color, sq)  # opponent attacks it
    ]
    if now_hanging:
        facts.append(f"leaves own {', '.join(now_hanging)} undefended")

    return facts


# ---------------------------------------------------------------------------
# Prompt formatting — shared between training and inference
# ---------------------------------------------------------------------------


def format_user_prompt(
    board_ascii_str: str,
    san: str,
    classification: str,
    eval_str: str,
    best_move: str = "",
    cp_loss: int = 0,
    candidates: list[str] | None = None,
    opponent_threats: list[str] | None = None,
    facts: list[str] | None = None,
    fen: str = "",
) -> str:
    """Build the user message for the chess coaching prompt.

    This function is the single source of truth for the user prompt format,
    used identically by the inference server and the training data pipeline.
    """
    best_line = (
        f"Engine's best move was: {best_move} (−{cp_loss} centipawns)\n" if cp_loss > 0 else ""
    )
    candidates_line = f"Engine's top candidates: {', '.join(candidates)}\n" if candidates else ""
    threats_line = (
        f"Opponent's threats if you passed: {', '.join(opponent_threats)}\n"
        if opponent_threats
        else ""
    )
    facts_line = (
        "Verified move facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n" if facts else ""
    )
    fen_line = f"FEN: {fen}\n" if fen else ""
    board_section = (
        f"Board position before the move:\n{board_ascii_str}\n{fen_line}\n"
        if board_ascii_str
        else ""
    )

    return (
        f"{board_section}"
        f"Move played: {san}\n"
        f"Classification: {classification}\n"
        f"Engine evaluation before move: {eval_str}\n"
        f"{best_line}"
        f"{candidates_line}"
        f"{threats_line}"
        f"{facts_line}"
        "\nExplain the chess idea behind this move in 2-3 sentences. "
        "Draw on opening theory, strategic plans, tactical motifs, or positional concepts as relevant. "
        "If verified move facts are listed, do not contradict them."
    )
