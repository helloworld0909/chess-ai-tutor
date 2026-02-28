"""Convert lines_30k.jsonl into the line-generator SFT format WITH pseudo-thinking.

Pseudo-thinking is generated deterministically from Stockfish + python-chess:
  1. Position assessment (cp eval → human label, or mate distance)
  2. Side to move and game phase (opening / middlegame / endgame)
  3. Top-N candidate moves with purpose annotations (from Stockfish multipv)
  4. A brief plan derived from the best line's first 2 moves
  5. One of several thinking-style templates, chosen pseudo-randomly per sample

This gives the model diverse thinking scaffolds to learn from, so GRPO can
later refine the quality of reasoning rather than bootstrap it from scratch.

Usage:
    uv run python data/pipeline/convert_lines_to_sft_thinking.py \\
        --input  data/processed/lines_30k.jsonl \\
        --output data/processed/lines_sft_thinking.jsonl \\
        --eval-split 0.05 \\
        --eval-output data/processed/lines_sft_thinking_eval.jsonl \\
        --workers 8 \\
        --depth 10
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import (
    LINE_GENERATOR_SYSTEM_PROMPT,
    board_ascii,
    format_line_generator_prompt,
    move_facts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", os.path.expanduser("~/.local/bin/stockfish"))


# ---------------------------------------------------------------------------
# Move purpose annotations (python-chess, no LLM)
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _move_purpose(board: chess.Board, move: chess.Move) -> str:
    """Return a short purpose annotation for a move (3-5 words)."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "make a move"

    pname = _PIECE_NAMES.get(piece.piece_type, "piece")
    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)

    # Castling
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
        return f"castle {side}"

    # Promotion
    if move.promotion:
        promoted = _PIECE_NAMES.get(move.promotion, "queen")
        return f"promote pawn to {promoted}"

    suffix = " with check" if gives_check else ""

    if is_capture:
        victim = board.piece_at(move.to_square)
        vname = _PIECE_NAMES.get(victim.piece_type, "piece") if victim else "piece"
        return f"capture {vname}{suffix}"

    # Piece-specific descriptors
    if piece.piece_type == chess.PAWN:
        if chess.square_rank(move.to_square) in (3, 4):  # ranks 4/5 (0-indexed)
            return f"advance pawn{suffix}" if not suffix else f"pawn advance{suffix}"
        return f"push pawn{suffix}"
    if piece.piece_type == chess.KNIGHT:
        return (
            f"develop knight{suffix}"
            if chess.square_rank(move.from_square) in (0, 7)
            else f"relocate knight{suffix}"
        )
    if piece.piece_type == chess.BISHOP:
        return (
            f"develop bishop{suffix}"
            if chess.square_rank(move.from_square) in (0, 7)
            else f"reposition bishop{suffix}"
        )
    if piece.piece_type == chess.ROOK:
        return f"activate rook{suffix}"
    if piece.piece_type == chess.QUEEN:
        return f"activate queen{suffix}"
    if piece.piece_type == chess.KING:
        return f"move king{suffix}"

    return f"move {pname}{suffix}"


# ---------------------------------------------------------------------------
# Stockfish helpers (synchronous, one engine per worker process)
# ---------------------------------------------------------------------------

_engine: Optional[chess.engine.SimpleEngine] = None


def _get_engine() -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        return _engine
    # Check if the engine process is still alive; restart if not
    try:
        _engine.ping()
    except Exception:
        try:
            _engine.close()
        except Exception:
            pass
        _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    return _engine


def _cp_to_label(cp: int, turn: chess.Color) -> str:
    """Convert centipawns (from White's perspective) to a human label."""
    # Flip if it's Black's turn (eval is always from White's perspective)
    if turn == chess.BLACK:
        cp = -cp
    # From the side-to-move's perspective
    if cp >= 200:
        return "winning for me"
    if cp >= 60:
        return "good for me"
    if cp >= -60:
        return "roughly equal"
    if cp >= -200:
        return "slightly worse"
    return "difficult position"


def _game_phase(board: chess.Board) -> str:
    piece_count = len(board.piece_map())
    if piece_count >= 28:
        return "opening"
    if piece_count >= 14:
        return "middlegame"
    return "endgame"


def _analyze_position(
    fen: str, depth: int = 10, multipv: int = 5
) -> tuple[int | None, list[tuple[str, str, int | None]]]:
    """Return (cp_score, [(san, purpose, cp), ...]) for top multipv moves.

    cp_score is from White's perspective (None if mate).
    Candidate tuples contain (san_move, purpose_annotation, cp_or_None).
    """
    engine = _get_engine()
    board = chess.Board(fen)

    try:
        result = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=multipv,
        )
    except Exception:
        return None, []

    candidates: list[tuple[str, str, int | None]] = []
    root_cp: int | None = None

    for i, info in enumerate(result):
        pv = info.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        score = info.get("score")

        # Root eval from the first (best) line
        cp: int | None = None
        if score is not None:
            if score.is_mate():
                if i == 0:
                    root_cp = None
            else:
                cp_val = score.white().score()
                if i == 0:
                    root_cp = cp_val
                cp = cp_val

        try:
            san = board.san(move)
        except Exception:
            continue
        purpose = _move_purpose(board, move)
        candidates.append((san, purpose, cp))

    return root_cp, candidates


# ---------------------------------------------------------------------------
# Pseudo-thinking templates (diverse styles)
# ---------------------------------------------------------------------------

_TEMPLATES = [
    # Template 0 — structured enumeration
    lambda ctx: (
        f"Let me assess the position. FEN: {ctx['fen']}\n"
        f"It is {ctx['turn_str']} to move ({ctx['phase']}).\n"
        f"Engine evaluation: {ctx['eval_str']}.\n"
        f"\nCandidate moves to consider:\n{ctx['candidates_str']}\n"
        f"\nI'll generate 5 lines exploring these ideas, making sure each starts "
        f"with a different candidate move for breadth."
    ),
    # Template 1 — reasoning through the eval first
    lambda ctx: (
        f"Position: {ctx['fen']}. {ctx['turn_str']} to move, {ctx['phase']}.\n"
        f"The position is {ctx['eval_str']}.\n"
        f"Top engine candidates: {ctx['candidates_inline']}.\n"
        f"Let me trace out 5 distinct continuations, verifying each move is legal "
        f"before writing it down."
    ),
    # Template 2 — concise plan-first
    lambda ctx: (
        f"FEN: {ctx['fen']} — {ctx['phase']}, {ctx['turn_str']} to move.\n"
        f"Eval: {ctx['eval_str']}. Best: {ctx['best_move_str']}.\n"
        f"I need 5 instructive lines. I'll start from the top candidates "
        f"({ctx['candidates_inline']}) and make sure each explores a different idea."
    ),
    # Template 3 — enumerate + legality check note
    lambda ctx: (
        f"Analysing position {ctx['fen']}.\n"
        f"Turn: {ctx['turn_str']}. Phase: {ctx['phase']}. Eval: {ctx['eval_str']}.\n"
        f"\nStockfish top moves:\n{ctx['candidates_str']}\n"
        f"\nFor each line I'll verify the sequence is legal and the response moves "
        f"are reasonable. Goal: 5 lines, each starting differently."
    ),
    # Template 4 — brief + tactical focus
    lambda ctx: (
        f"{ctx['turn_str'].capitalize()} to move. {ctx['phase'].capitalize()}. {ctx['eval_str']}.\n"
        f"Candidates: {ctx['candidates_inline']}.\n"
        f"Checking for tactics first, then strategic lines. Will produce 5 distinct lines."
    ),
]


def _build_thinking(
    fen: str,
    root_cp: int | None,
    candidates: list[tuple[str, str, int | None]],
    rng: random.Random,
) -> str:
    """Build a pseudo-thinking block for the given position."""
    board = chess.Board(fen)
    turn_str = "white" if board.turn == chess.WHITE else "black"
    phase = _game_phase(board)

    if root_cp is None:
        eval_str = "a forced mate sequence exists"
    else:
        eval_str = _cp_to_label(root_cp, board.turn)

    # Format candidate list (bullet style)
    if candidates:
        lines = []
        for san, purpose, cp in candidates[:5]:
            cp_note = f" ({cp:+d} cp)" if cp is not None else ""
            lines.append(f"  - {san} ({purpose}){cp_note}")
        candidates_str = "\n".join(lines)
        candidates_inline = ", ".join(f"{san} ({purpose})" for san, purpose, _ in candidates[:4])
        best_move_str = f"{candidates[0][0]} ({candidates[0][1]})"
    else:
        candidates_str = "  (no candidates available)"
        candidates_inline = "unknown"
        best_move_str = "unknown"

    ctx = {
        "fen": fen,
        "turn_str": turn_str,
        "phase": phase,
        "eval_str": eval_str,
        "candidates_str": candidates_str,
        "candidates_inline": candidates_inline,
        "best_move_str": best_move_str,
    }

    template = rng.choice(_TEMPLATES)
    return template(ctx)


# ---------------------------------------------------------------------------
# Per-sample conversion (runs in worker process)
# ---------------------------------------------------------------------------


def _consensus_eval(lines: list[str]) -> str:
    from collections import Counter

    labels: list[str] = []
    for line in lines:
        if "| eval:" in line:
            labels.append(line.split("| eval:")[-1].strip().lower())
    if not labels:
        return "equal"
    return Counter(labels).most_common(1)[0][0]


def _wrap_lines(raw_lines: list[str]) -> str:
    return "\n".join(f"<line>{l.strip()}</line>" for l in raw_lines)


def convert_sample(args_tuple: tuple) -> dict | None:
    """Convert one raw sample to SFT format with pseudo-thinking.

    Runs in a worker process — creates its own Stockfish engine.
    """
    sample, depth, seed_offset = args_tuple
    fen = sample["fen"]
    move_uci = sample.get("move_uci", "")
    move_san = sample["move_san"]

    rng = random.Random(hash(fen + move_san) + seed_offset)

    try:
        board = chess.Board(fen)
        board_str = board_ascii(board)
    except Exception:
        return None

    facts: list[str] = []
    board_after_str = ""
    fen_after = ""
    try:
        move = board.parse_san(move_san)
        facts = move_facts(board, move)
        board_after = board.copy()
        board_after.push(move)
        board_after_str = board_ascii(board_after)
        fen_after = board_after.fen()
    except Exception:
        pass

    eval_str = _consensus_eval(sample["lines"])
    user_content = format_line_generator_prompt(
        board_str, fen, move_san, eval_str, facts, board_after_str, fen_after
    )

    # Run Stockfish for this position
    try:
        root_cp, candidates = _analyze_position(fen, depth=depth, multipv=5)
    except Exception:
        root_cp, candidates = None, []

    thinking = _build_thinking(fen, root_cp, candidates, rng)
    assistant_content = f"<think>\n{thinking}\n</think>\n" + _wrap_lines(sample["lines"])

    return {
        "messages": [
            {"role": "system", "content": LINE_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "lichess_lines_sft_thinking",
            "fen": fen,
            "move_san": move_san,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/lines_30k.jsonl")
    parser.add_argument("--output", default="data/processed/lines_sft_thinking.jsonl")
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--eval-output", default="data/processed/lines_sft_thinking_eval.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        log.error("Input not found: %s", src)
        sys.exit(1)

    # Load
    all_samples: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                if rec.get("metadata", {}).get("source") == "lichess_lines" and rec.get("lines"):
                    all_samples.append(rec)
            except json.JSONDecodeError:
                pass

    log.info("Loaded %d samples", len(all_samples))

    # Split
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_eval = max(1, int(len(all_samples) * args.eval_split))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    log.info("Split: %d train / %d eval", len(train_samples), len(eval_samples))

    # Tag each sample with its split so we can route output in a single pool run
    train_tagged = [(s, args.depth, args.seed, "train") for s in train_samples]
    eval_tagged = [(s, args.depth, args.seed, "eval") for s in eval_samples]
    all_tagged = train_tagged + eval_tagged

    train_out = Path(args.output)
    eval_out = Path(args.eval_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "eval": 0}
    failed = {"train": 0, "eval": 0}
    total = len(all_tagged)

    ctx = multiprocessing.get_context("spawn")
    with (
        train_out.open("w", encoding="utf-8") as f_train,
        eval_out.open("w", encoding="utf-8") as f_eval,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool,
    ):
        # convert_sample takes (sample, depth, seed) — strip the split tag before submitting
        futures = {
            pool.submit(convert_sample, (s, depth, seed)): split
            for s, depth, seed, split in all_tagged
        }

        for i, fut in enumerate(as_completed(futures)):
            split = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                log.warning("Sample failed: %s", e)
                result = None

            if result is not None:
                fout = f_train if split == "train" else f_eval
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                written[split] += 1
            else:
                failed[split] += 1

            if (i + 1) % 500 == 0:
                log.info(
                    "  %d / %d done  train(w=%d,f=%d)  eval(w=%d,f=%d)",
                    i + 1,
                    total,
                    written["train"],
                    failed["train"],
                    written["eval"],
                    failed["eval"],
                )

    log.info(
        "Done. train=%d eval=%d (failed train=%d eval=%d)",
        written["train"],
        written["eval"],
        failed["train"],
        failed["eval"],
    )


if __name__ == "__main__":
    main()
