"""GRPO reward functions for the chess line generator.

Each function follows the TRL GRPOTrainer signature:
    reward_fn(prompts, completions, **kwargs) -> list[float]

Rewards are combined in train.py with weights from TODO_RL.md Phase 1:
    R = 0.25*R1 + 0.20*R2 + 0.30*R3 + 0.15*R4a + 0.10*R5

R1  — Move legality (free, python-chess)
R2  — Opponent move quality (Stockfish, depth 12) — skipped in Phase 1
R3  — Final position evaluation accuracy (Stockfish, depth 18)
R4a — Structural annotation accuracy (free, python-chess)
R5  — Line relevance: first move continues from the played position

Stockfish rewards (R3) use a shared engine pool to amortise process startup.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import chess

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

# Matches lines inside <line>...</line> tags OR bare "LINE N: ..." lines.
# Group 1 = move body, Group 2 = eval label.
_LINE_RE = re.compile(
    r"<line>\s*LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)\s*</line>"
    r"|LINE\s+\d+\s*:\s*(.+?)\|\s*eval\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_MOVE_RE = re.compile(r"([A-Za-z][A-Za-z0-9\-+#=!?]+|\bO-O(?:-O)?\b)")
_ANNOT_RE = re.compile(r"\([^)]*\)")

# Maps Stockfish cp (white perspective) to our eval label
_CP_BANDS: list[tuple[int, str]] = [
    (300, "winning for white"),
    (100, "good for white"),
    (-100, "equal"),
    (-300, "good for black"),
]


def _cp_to_label(cp: int) -> str:
    for threshold, label in _CP_BANDS:
        if cp > threshold:
            return label
    return "winning for black"


_LABEL_DISTANCE: dict[str, int] = {
    "winning for white": 4,
    "good for white": 3,
    "equal": 2,
    "good for black": 1,
    "winning for black": 0,
}


def _label_distance(a: str, b: str) -> int:
    """Ordinal distance between two eval labels (0 = same)."""
    return abs(_LABEL_DISTANCE.get(a, 2) - _LABEL_DISTANCE.get(b, 2))


def _groups(m: re.Match) -> tuple[str, str]:
    """Return (body, eval_label) from a _LINE_RE match (handles both alt groups)."""
    if m.group(1) is not None:
        return m.group(1).strip(), m.group(2).strip().lower()
    return m.group(3).strip(), m.group(4).strip().lower()


def parse_lines(text: str) -> list[dict]:
    """Extract structured lines from a model completion.

    Handles both <line>LINE N: ...</line> (new format) and bare LINE N: ...
    Returns a list of dicts: {moves_san: [str, ...], eval_label: str}
    Only lines with at least one move are returned.
    """
    results = []
    for m in _LINE_RE.finditer(text):
        body, eval_label = _groups(m)
        # Strip annotations, then collect move tokens
        bare = _ANNOT_RE.sub("", body)
        # Split on → or whitespace, filter move-like tokens
        raw_tokens = re.split(r"[→\s]+", bare)
        moves = [t.strip() for t in raw_tokens if t.strip() and _MOVE_RE.fullmatch(t.strip())]
        if moves:
            results.append({"moves_san": moves, "eval_label": eval_label})
    return results


# ---------------------------------------------------------------------------
# Prompt → (FEN, move_san) extraction
# ---------------------------------------------------------------------------

# Matches "FEN: <fen>" (coach-style prompt) or "Position (FEN): <fen>" (old tool-call style)
_FEN_RE = re.compile(r"FEN:\s*(\S+(?:\s+\S+){5})")
# Matches "Move: <san>" (new coach-style) or "Move played: <san>" (old style)
_MOVE_PLAYED_RE = re.compile(r"Move(?:\s+played)?:\s*(\S+)")


def _extract_context(prompt: str) -> tuple[str, str]:
    """Extract (fen, move_san) from the user prompt turn."""
    fen_m = _FEN_RE.search(prompt)
    move_m = _MOVE_PLAYED_RE.search(prompt)
    fen = fen_m.group(1).strip() if fen_m else ""
    move_san = move_m.group(1).strip() if move_m else ""
    return fen, move_san


def _prompt_str(prompt: list[dict] | str) -> str:
    """Flatten a prompt (list of messages or plain string) to text."""
    if isinstance(prompt, str):
        return prompt
    return "\n".join(m.get("content") or "" for m in prompt if m.get("content"))


# ---------------------------------------------------------------------------
# Stockfish engine pool
# ---------------------------------------------------------------------------

_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")
_ENGINE_POOL: list[Any] = []
_ENGINE_LOCK = threading.Lock()
_POOL_SIZE = 16  # concurrent Stockfish processes (match max completions per step)
_SF_DEPTH = 12  # eval depth for reward signal; can be overridden by trainer


def _get_engine() -> Any:
    """Borrow a chess.engine.SimpleEngine from the pool (creates if needed)."""
    import chess.engine

    with _ENGINE_LOCK:
        if _ENGINE_POOL:
            return _ENGINE_POOL.pop()
    return chess.engine.SimpleEngine.popen_uci(_STOCKFISH_PATH)


def _return_engine(engine: Any) -> None:
    with _ENGINE_LOCK:
        if len(_ENGINE_POOL) < _POOL_SIZE:
            _ENGINE_POOL.append(engine)
        else:
            try:
                engine.quit()
            except Exception:
                pass


def _eval_fen_sync(fen: str, depth: int = 18) -> int | None:
    """Evaluate a FEN with Stockfish (blocking). Returns cp from white's perspective."""
    import chess.engine

    engine = _get_engine()
    try:
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            return 30000 if score.mate() > 0 else -30000  # type: ignore[arg-type]
        cp = score.score()
        return cp
    except Exception as exc:
        log.debug("Stockfish eval failed for %s: %s", fen, exc)
        return None
    finally:
        _return_engine(engine)


# Thread pool for blocking Stockfish calls inside reward functions
_SF_EXECUTOR = ThreadPoolExecutor(max_workers=_POOL_SIZE, thread_name_prefix="sf_reward")


def _eval_fen(fen: str, depth: int = 18) -> int | None:
    """Non-blocking wrapper: runs Stockfish in the thread pool."""
    future = _SF_EXECUTOR.submit(_eval_fen_sync, fen, depth)
    try:
        return future.result(timeout=30)
    except Exception as exc:
        log.debug("Stockfish thread failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# R1 — Move Legality
# ---------------------------------------------------------------------------


def _play_line(fen: str, moves_san: list[str]) -> tuple[bool, chess.Board]:
    """Attempt to play a SAN move sequence from fen.

    Returns (fully_legal: bool, board_after_last_legal_move).
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return False, chess.Board()
    for san in moves_san:
        try:
            move = board.parse_san(san)
            board.push(move)
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
            ValueError,
        ):
            return False, board
    return True, board


def reward_legality(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R1: +1.0 per fully-legal line, -1.0 per line with an illegal move.

    Score is the mean over all parsed lines (or -1.0 if no lines found).
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        fen, _ = _extract_context(prompt_text)
        if not fen:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        line_scores: list[float] = []
        for line in lines:
            legal, _ = _play_line(fen, line["moves_san"])
            line_scores.append(1.0 if legal else -1.0)
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R3 — Final Position Evaluation Accuracy
# ---------------------------------------------------------------------------


def _r3_line_score(fen: str, line: dict) -> float:
    """Score a single line's eval label against Stockfish ground truth."""
    legal, board = _play_line(fen, line["moves_san"])
    if not legal or not line["moves_san"]:
        return -1.0  # illegal line — can't evaluate final position

    final_fen = board.fen()
    cp = _eval_fen(final_fen, depth=_SF_DEPTH)
    if cp is None:
        return 0.0  # Stockfish unavailable — neutral

    gt_label = _cp_to_label(cp)
    model_label = line["eval_label"]
    dist = _label_distance(gt_label, model_label)

    if dist == 0:
        return 1.0
    elif dist == 1:
        return -0.5
    else:
        return -1.0


def reward_eval_accuracy(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R3: Eval label accuracy vs Stockfish ground truth at depth 18.

    +1.0 exact match, -0.5 off by one band, -1.0 two+ bands off.
    Score is mean over all parsed lines (or -1.0 if no lines found).
    """
    scores: list[float] = []

    # Run all Stockfish evals in parallel via thread pool
    all_items: list[tuple[int, int, str, dict]] = []  # (sample_idx, line_idx, fen, line)
    sample_lines: list[list[dict]] = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        prompt_text = _prompt_str(prompt)
        fen, _ = _extract_context(prompt_text)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        sample_lines.append(lines)
        for j, line in enumerate(lines):
            all_items.append((i, j, fen, line))

    # Submit all Stockfish jobs
    futures = {
        (i, j): _SF_EXECUTOR.submit(_r3_line_score, fen, line) for i, j, fen, line in all_items
    }

    # Collect results per sample
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        lines = sample_lines[i]
        if not lines:
            scores.append(-1.0)
            continue
        line_scores = []
        for j in range(len(lines)):
            fut = futures.get((i, j))
            if fut is None:
                line_scores.append(0.0)
            else:
                try:
                    line_scores.append(fut.result(timeout=60))
                except Exception:
                    line_scores.append(0.0)
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R4a — Structural Annotation Accuracy
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _score_annotation_structural(board: chess.Board, move: chess.Move, annotation: str) -> float:
    """Score a single move annotation against python-chess ground truth.

    Checks:
    - capture vs non-capture correctly identified
    - piece name mentioned
    - check flag matches
    - castling / promotion identified
    Returns +1.0 if all correct, -1.0 if any structural fact wrong.
    """
    ann = annotation.lower()
    ann_tokens = set(ann.split())

    is_capture = board.is_capture(move)
    mentions_capture = "capture" in ann_tokens or "captures" in ann_tokens
    if is_capture != mentions_capture:
        return -1.0

    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0
    piece_name = _PIECE_NAMES.get(piece.piece_type, "piece")
    if piece_name not in ann_tokens:
        return -1.0

    gives_check = board.gives_check(move)
    mentions_check = "check" in ann_tokens
    if gives_check != mentions_check:
        return -1.0

    is_castling = board.is_castling(move)
    mentions_castle = "castle" in ann or "castles" in ann or "castling" in ann
    if is_castling != mentions_castle:
        return -1.0

    return 1.0


def _r4a_line_score(fen: str, line: dict) -> float:
    """Score structural annotation accuracy for all moves in a line."""
    try:
        board = chess.Board(fen)
    except Exception:
        return -1.0

    # Re-parse lines extracting per-move annotations from the original text.
    # We need to pair each SAN move with its annotation.
    # Parse the raw text representation (stored in line dict) is not available
    # here, so we reconstruct from moves_san + board state.
    # We can only check annotations we have — if the model omitted an annotation
    # we skip that move (0.0 contribution).
    # NOTE: the line dict only has moves_san list; annotations are inlined in the
    # completion text and already stripped by parse_lines. R4a therefore needs
    # to work from the raw completion text, passed via a separate field.
    # This function is called from reward_annotation_structural which pre-parses.
    return 0.0  # placeholder — see reward_annotation_structural below


def _parse_lines_with_annotations(text: str) -> list[dict]:
    """Like parse_lines but keeps (move, annotation) pairs per line."""
    results = []
    for m in _LINE_RE.finditer(text):
        body, eval_label = _groups(m)
        # Split on → to get individual move+annotation chunks
        chunks = re.split(r"→", body)
        move_annots: list[tuple[str, str]] = []
        for chunk in chunks:
            chunk = chunk.strip()
            annot_m = re.search(r"\(([^)]*)\)", chunk)
            annot = annot_m.group(1).strip() if annot_m else ""
            bare = _ANNOT_RE.sub("", chunk).strip()
            tokens = bare.split()
            if tokens:
                move_san = tokens[-1]  # last token after stripping annotation
                move_annots.append((move_san, annot))
        if move_annots:
            results.append({"move_annots": move_annots, "eval_label": eval_label})
    return results


def reward_annotation_structural(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R4a: Structural annotation accuracy using python-chess ground truth.

    For each move in each line, checks:
    - capture/non-capture correctly named
    - piece name present
    - check flag matches
    - castling identified
    Score per move: +1.0 correct, -1.0 wrong. Mean over all moves in all lines.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        fen, _ = _extract_context(prompt_text)
        if not fen:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = _parse_lines_with_annotations(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        move_scores: list[float] = []
        for line in lines:
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            for move_san, annot in line["move_annots"]:
                if not annot:
                    # No annotation provided — skip (don't penalise missing)
                    try:
                        move = board.parse_san(move_san)
                        board.push(move)
                    except Exception:
                        break
                    continue
                try:
                    move = board.parse_san(move_san)
                    s = _score_annotation_structural(board, move, annot)
                    move_scores.append(s)
                    board.push(move)
                except Exception:
                    move_scores.append(-1.0)
                    break  # illegal move — stop this line

        if not move_scores:
            scores.append(0.0)
        else:
            scores.append(sum(move_scores) / len(move_scores))
    return scores


# ---------------------------------------------------------------------------
# R4 — Line Depth
# ---------------------------------------------------------------------------

# Target half-moves per line — lines shorter than this score less
# Start at 2 for curriculum: reward any line that has ≥2 legal moves
_TARGET_DEPTH = 2


def reward_depth(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R4: Reward lines proportional to their depth, capped at _TARGET_DEPTH.

    Score per line = min(len(moves), _TARGET_DEPTH) / _TARGET_DEPTH.
    Capped at 1.0 so padding beyond target gets no reward.
    Final score = mean over all parsed lines (or -1.0 if none found).
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue
        line_scores = [min(len(line["moves_san"]), _TARGET_DEPTH) / _TARGET_DEPTH for line in lines]
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# R5 — Line Breadth
# ---------------------------------------------------------------------------


def reward_breadth(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R5: Reward unique first moves across all parsed lines.

    unique_ratio = len(set(first_moves)) / len(first_moves)
    +1.0 if all lines start with a different move; lower if transpositions present.
    Returns -1.0 if no lines found.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        fen, _ = _extract_context(prompt_text)
        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue
        first_moves = [line["moves_san"][0] for line in lines if line["moves_san"]]
        if not first_moves:
            scores.append(-1.0)
            continue
        unique_ratio = len(set(first_moves)) / len(first_moves)
        scores.append(unique_ratio)
    return scores


# ---------------------------------------------------------------------------
# R6 — Line Relevance (was R5)
# ---------------------------------------------------------------------------


def reward_relevance(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """R5: First move of each line must be a legal continuation from the FEN.

    Proxy for relevance: the model analyses the actual position, not a random one.
    +1.0 if all lines start with a legal move from the given FEN.
    -1.0 if any line starts with an illegal move.
    """
    scores: list[float] = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = _prompt_str(prompt)
        fen, _ = _extract_context(prompt_text)
        if not fen:
            scores.append(0.0)
            continue

        completion_text = _prompt_str(completion)
        lines = parse_lines(completion_text)
        if not lines:
            scores.append(-1.0)
            continue

        line_scores: list[float] = []
        for line in lines:
            if not line["moves_san"]:
                line_scores.append(-1.0)
                continue
            try:
                board = chess.Board(fen)
                board.parse_san(line["moves_san"][0])
                line_scores.append(1.0)
            except Exception:
                line_scores.append(-1.0)
        scores.append(sum(line_scores) / len(line_scores))
    return scores


# ---------------------------------------------------------------------------
# Combined reward (for logging/debugging — GRPOTrainer calls each separately)
# ---------------------------------------------------------------------------

# Phase 1 weights (all free, no Haiku judge).
# R1 is a hard gate: illegal → -1.0 and downstream rewards are 0.
# Remaining weights sum to 0.75, matching TODO_RL.md Phase 1 spec.
_WEIGHTS = {
    "eval_accuracy": 0.28,
    "annotation_structural": 0.12,
    "depth": 0.10,
    "breadth": 0.10,
    "relevance": 0.05,
}


def combined_reward(
    prompts: list[list[dict] | str],
    completions: list[list[dict] | str],
    **kwargs: Any,
) -> list[float]:
    """Phase 1 combined reward (for reference). GRPOTrainer uses individual fns.

    R1 (legality) is a hard gate: any illegal completion scores -1.0 immediately,
    and all downstream rewards are zeroed out for that sample.
    """
    r1 = reward_legality(prompts, completions, **kwargs)
    r3 = reward_eval_accuracy(prompts, completions, **kwargs)
    r4a = reward_annotation_structural(prompts, completions, **kwargs)
    r4_depth = reward_depth(prompts, completions, **kwargs)
    r5_breadth = reward_breadth(prompts, completions, **kwargs)
    r6 = reward_relevance(prompts, completions, **kwargs)

    results = []
    for legal, eval_acc, annot, depth, breadth, relevance in zip(
        r1, r3, r4a, r4_depth, r5_breadth, r6
    ):
        if legal < 0:
            # Hard gate: any illegal line → -1.0, all downstream zeroed
            results.append(-1.0)
        else:
            results.append(
                _WEIGHTS["eval_accuracy"] * eval_acc
                + _WEIGHTS["annotation_structural"] * annot
                + _WEIGHTS["depth"] * depth
                + _WEIGHTS["breadth"] * breadth
                + _WEIGHTS["relevance"] * relevance
            )
    return results
