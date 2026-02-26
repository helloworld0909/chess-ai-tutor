"""Shared chess analysis utilities.

Used by both the training data pipeline and the inference server to compute
position features: CCT moves, move trees, position context, and score formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import chess

if TYPE_CHECKING:
    from chess_mcp.stockfish import Stockfish

# ---------------------------------------------------------------------------
# Score formatting
# ---------------------------------------------------------------------------


def format_score(value: int, is_mate: bool) -> str:
    """Format a centipawn or mate score for display (e.g. '+0.35' or 'M3')."""
    if is_mate:
        return f"M{abs(value)}" if value > 0 else f"-M{abs(value)}"
    return f"{value / 100:+.2f}"


# ---------------------------------------------------------------------------
# CCT (Checks / Captures / Threats)
# ---------------------------------------------------------------------------

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def compute_cct(board: chess.Board) -> dict[str, list[str]]:
    """Compute Check/Capture/Threat moves available to the side to move.

    - checks:   legal moves that immediately give check
    - captures: legal moves that capture an opponent piece
    - threats:  non-check, non-capture moves that attack an undefended
                opponent piece of strictly higher value than the attacker
    """
    side = board.turn
    opp = not side

    checks: list[str] = []
    captures: list[str] = []
    threats: list[str] = []

    for move in board.legal_moves:
        is_check = board.gives_check(move)
        is_capture = board.is_capture(move)

        if is_check:
            checks.append(board.san(move))
        elif is_capture:
            captures.append(board.san(move))
        else:
            attacker_piece = board.piece_at(move.from_square)
            attacker_val = _PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

            board.push(move)
            for sq in chess.SQUARES:
                target = board.piece_at(sq)
                if target and target.color == opp:
                    target_val = _PIECE_VALUES.get(target.piece_type, 0)
                    if target_val > attacker_val and board.is_attacked_by(side, sq):
                        if not board.is_attacked_by(opp, sq):
                            threats.append(board.peek().uci())
                            break
            board.pop()

    threat_sans: list[str] = []
    for uci in threats:
        try:
            threat_sans.append(board.san(chess.Move.from_uci(uci)))
        except Exception:
            pass

    return {"checks": checks[:5], "captures": captures[:5], "threats": threat_sans[:5]}


# ---------------------------------------------------------------------------
# Move tree
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """One node in the 3-level post-move candidate tree.

    L1: opponent's top replies after the played move.
    L2: player's top replies to each L1 move.
    L3: opponent's top counter-replies (eval only, no CCT).
    """

    move_san: str
    eval_str: str
    # CCT for the side to move AFTER this node's move was played.
    # Populated for L1 (your options) and L2 (opponent options); empty for L3.
    cct: dict[str, list[str]] = field(default_factory=dict)
    children: list[TreeNode] = field(default_factory=list)


def tree_node_to_dict(node: TreeNode) -> dict[str, Any]:
    """Serialize a TreeNode to a JSON-serializable dict."""
    return {
        "move": node.move_san,
        "eval": node.eval_str,
        "cct": node.cct,
        "children": [tree_node_to_dict(c) for c in node.children],
    }


def tree_node_from_dict(d: dict[str, Any]) -> TreeNode:
    """Deserialize a TreeNode from a dict."""
    return TreeNode(
        move_san=d["move"],
        eval_str=d["eval"],
        cct=d.get("cct", {}),
        children=[tree_node_from_dict(c) for c in d.get("children", [])],
    )


async def build_move_tree(
    engine: Stockfish,
    board_after: chess.Board,
    depth: int = 14,
    width: int = 3,
) -> list[TreeNode]:
    """Build a 3-level response tree rooted at *board_after*.

    L1: opponent's top *width* replies (+ CCT for you after each).
    L2: your top *width* replies to each L1 (+ CCT for opponent after each).
    L3: opponent's top *width* counter-replies (eval only, no CCT).

    Total Stockfish calls: 1 + width + width² = 13 (width=3).
    Returns the list of L1 TreeNodes; each has .children (L2), each of which
    has .children (L3).
    """
    if board_after.is_game_over():
        return []

    # L1: analyze opponent's replies
    l1_analysis = await engine.analyze(board_after.fen(), depth=depth, multipv=width)
    l1_nodes: list[TreeNode] = []

    for l1_line in l1_analysis.lines[:width]:
        try:
            l1_move = chess.Move.from_uci(l1_line.best_move)
            l1_san = board_after.san(l1_move)
            l1_sc = l1_line.score
            l1_eval = format_score(
                l1_sc.mate_in if l1_sc.mate_in is not None else (l1_sc.centipawns or 0),
                l1_sc.mate_in is not None,
            )
        except Exception:
            continue

        board_l1 = board_after.copy()
        board_l1.push(l1_move)

        # CCT available to the player after opponent's L1 reply
        l1_cct = compute_cct(board_l1)

        l2_nodes: list[TreeNode] = []
        if not board_l1.is_game_over():
            l2_analysis = await engine.analyze(board_l1.fen(), depth=depth, multipv=width)

            for l2_line in l2_analysis.lines[:width]:
                try:
                    l2_move = chess.Move.from_uci(l2_line.best_move)
                    l2_san = board_l1.san(l2_move)
                    l2_sc = l2_line.score
                    l2_eval = format_score(
                        l2_sc.mate_in if l2_sc.mate_in is not None else (l2_sc.centipawns or 0),
                        l2_sc.mate_in is not None,
                    )
                except Exception:
                    continue

                board_l2 = board_l1.copy()
                board_l2.push(l2_move)

                # CCT available to opponent after the player's L2 reply
                l2_cct = compute_cct(board_l2)

                l3_nodes: list[TreeNode] = []
                if not board_l2.is_game_over():
                    l3_analysis = await engine.analyze(board_l2.fen(), depth=depth, multipv=width)

                    for l3_line in l3_analysis.lines[:width]:
                        try:
                            l3_move = chess.Move.from_uci(l3_line.best_move)
                            l3_san = board_l2.san(l3_move)
                            l3_sc = l3_line.score
                            l3_eval = format_score(
                                l3_sc.mate_in
                                if l3_sc.mate_in is not None
                                else (l3_sc.centipawns or 0),
                                l3_sc.mate_in is not None,
                            )
                        except Exception:
                            continue
                        l3_nodes.append(TreeNode(move_san=l3_san, eval_str=l3_eval))

                l2_nodes.append(
                    TreeNode(
                        move_san=l2_san,
                        eval_str=l2_eval,
                        cct=l2_cct,
                        children=l3_nodes,
                    )
                )

        l1_nodes.append(
            TreeNode(
                move_san=l1_san,
                eval_str=l1_eval,
                cct=l1_cct,
                children=l2_nodes,
            )
        )

    return l1_nodes


# ---------------------------------------------------------------------------
# Position context (positional/structural + game-phase)
# ---------------------------------------------------------------------------

_MATERIAL_VALUES = {
    chess.QUEEN: 9,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.PAWN: 1,
}


def _passed_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Return square names of passed pawns for *color*."""
    opp = not color
    result = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        adj_files = [af for af in (f - 1, f, f + 1) if 0 <= af <= 7]
        is_passed = True
        for opp_sq in board.pieces(chess.PAWN, opp):
            of = chess.square_file(opp_sq)
            if of in adj_files:
                or_ = chess.square_rank(opp_sq)
                if color == chess.WHITE and or_ > r:
                    is_passed = False
                    break
                if color == chess.BLACK and or_ < r:
                    is_passed = False
                    break
        if is_passed:
            result.append(chess.square_name(sq))
    return result


def _doubled_files(board: chess.Board, color: chess.Color) -> list[str]:
    """Return file letters that have 2+ pawns of *color*."""
    files: list[int] = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]
    return [chr(ord("a") + f) for f in set(files) if files.count(f) >= 2]


def _isolated_pawns(board: chess.Board, color: chess.Color) -> list[str]:
    """Return square names of isolated pawns for *color*."""
    files: list[int] = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]
    result = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        if not any(af in files for af in (f - 1, f + 1) if 0 <= af <= 7):
            result.append(chess.square_name(sq))
    return result


def _file_openness(
    board: chess.Board,
) -> tuple[list[str], list[str], list[str]]:
    """Return (open_files, half_open_for_white, half_open_for_black) as file-letter lists."""
    w_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)}
    b_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)}
    open_files, half_white, half_black = [], [], []
    for f in range(8):
        fl = chr(ord("a") + f)
        w, b = f in w_files, f in b_files
        if not w and not b:
            open_files.append(fl)
        elif not w:
            half_white.append(fl)
        elif not b:
            half_black.append(fl)
    return open_files, half_white, half_black


def _piece_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count total attack squares for all non-pawn, non-king pieces of *color*."""
    total = 0
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in board.pieces(pt, color):
            total += len(board.attacks(sq))
    return total


def _king_shield(
    board: chess.Board,
    color: chess.Color,
    open_files: list[str],
    half_open: list[str],
) -> tuple[int, list[str]]:
    """Return (shield_pawn_count, open_file_letters_near_king) for *color*."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0, []
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    shield_rank = kr + 1 if color == chess.WHITE else kr - 1
    adj_files = [af for af in (kf - 1, kf, kf + 1) if 0 <= af <= 7]

    shield = 0
    if 0 <= shield_rank <= 7:
        for af in adj_files:
            sq = chess.square(af, shield_rank)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                shield += 1

    nearby_open = [
        chr(ord("a") + af)
        for af in adj_files
        if chr(ord("a") + af) in open_files or chr(ord("a") + af) in half_open
    ]
    return shield, nearby_open


def compute_position_context(board: chess.Board) -> dict[str, Any]:
    """Compute positional, structural, and game-phase context for a position.

    Returns a dict with keys:
      game_phase, move_number,
      material_white, material_black, material_balance,
      passed_pawns_white, passed_pawns_black,
      doubled_files_white, doubled_files_black,
      isolated_pawns_white, isolated_pawns_black,
      open_files, half_open_white, half_open_black,
      mobility_white, mobility_black,
      king_shield_white, king_shield_black,
      open_near_king_white, open_near_king_black.
    """
    # Material
    mat: dict[chess.Color, int] = {chess.WHITE: 0, chess.BLACK: 0}
    for color in (chess.WHITE, chess.BLACK):
        for pt, val in _MATERIAL_VALUES.items():
            mat[color] += len(board.pieces(pt, color)) * val

    # Game phase (based on non-pawn/non-king material present)
    minor_major = sum(
        len(board.pieces(pt, color)) * val
        for color in (chess.WHITE, chess.BLACK)
        for pt, val in _MATERIAL_VALUES.items()
        if pt != chess.PAWN
    )
    # Full set = 2*9 + 4*5 + 4*3 + 4*3 = 62
    if minor_major > 48:
        phase = "Opening"
    elif minor_major > 20:
        phase = "Middlegame"
    else:
        phase = "Endgame"

    # Pawn structure
    pp_w = _passed_pawns(board, chess.WHITE)
    pp_b = _passed_pawns(board, chess.BLACK)
    dbl_w = _doubled_files(board, chess.WHITE)
    dbl_b = _doubled_files(board, chess.BLACK)
    iso_w = _isolated_pawns(board, chess.WHITE)
    iso_b = _isolated_pawns(board, chess.BLACK)

    # File openness
    open_files, half_w, half_b = _file_openness(board)

    # Mobility
    mob_w = _piece_mobility(board, chess.WHITE)
    mob_b = _piece_mobility(board, chess.BLACK)

    # King safety
    shield_w, open_near_w = _king_shield(board, chess.WHITE, open_files, half_w)
    shield_b, open_near_b = _king_shield(board, chess.BLACK, open_files, half_b)

    return {
        "game_phase": phase,
        "move_number": board.fullmove_number,
        "material_white": mat[chess.WHITE],
        "material_black": mat[chess.BLACK],
        "material_balance": mat[chess.WHITE] - mat[chess.BLACK],
        "passed_pawns_white": pp_w,
        "passed_pawns_black": pp_b,
        "doubled_files_white": dbl_w,
        "doubled_files_black": dbl_b,
        "isolated_pawns_white": iso_w,
        "isolated_pawns_black": iso_b,
        "open_files": open_files,
        "half_open_white": half_w,
        "half_open_black": half_b,
        "mobility_white": mob_w,
        "mobility_black": mob_b,
        "king_shield_white": shield_w,
        "king_shield_black": shield_b,
        "open_near_king_white": open_near_w,
        "open_near_king_black": open_near_b,
    }
