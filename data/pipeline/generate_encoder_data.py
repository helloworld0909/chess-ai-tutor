"""Generate (fen_before, move_uci, cp_diff_scaled) triples for encoder pre-training.

Source: Lichess/fishnet-evals — 34.5B positions with per-move Stockfish evaluations.
No Stockfish needed — evaluations are pre-computed.

Each row in fishnet-evals:
    fen:  position BEFORE the move (cp is eval of this position, White's perspective)
    cp:   centipawn eval of fen (White's perspective); None if mate on board
    mate: mate distance if applicable
    move: UCI move actually played by the human

Sequential rows within a game let us compute:
    cp_before = row[i].cp
    cp_after  = row[i+1].cp  (next row's eval, which is the position after the move)

We validate continuity: applying row[i].move to row[i].fen must yield row[i+1].fen.
Rows that fail this check are treated as game boundaries (silently skipped).

Label: cp_diff_scaled = tanh((cp_after - cp_before) * side_sign / 600)
    side_sign = +1 if White moved, -1 if Black moved
    → positive = good move for the mover, negative = bad move

Static augmentation (--static-ratio):
    For every N real move records, one synthetic "static position" record is
    interleaved using the same FEN, move_uci=null, and label 0.0.
    This teaches the encoder the <position> (before==after) identity convention,
    where the expected cp diff is zero by construction.

Usage:
    uv run python data/pipeline/generate_encoder_data.py \\
        --output data/processed/encoder_pretrain.jsonl \\
        --limit 100000000 \\
        --min-ply 4 \\
        --max-ply 80 \\
        --static-ratio 0.01

Output record (move):
    {"fen": "<FEN before>", "move_uci": "<uci>", "cp_diff_scaled": 0.12}
Output record (static):
    {"fen": "<FEN>", "move_uci": null, "cp_diff_scaled": 0.0}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import chess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_HF_REPO = "Lichess/fishnet-evals"
_CP_CLAMP = 2000  # clamp raw cp before tanh scaling (avoids outlier distortion)
_CP_SCALE = 600  # tanh denominator: 600cp ≈ winning → ~0.93
_MATE_CP = _CP_CLAMP  # effective cp for mate positions (maps to tanh ≈ ±0.964)


def _effective_cp(cp: int | None, mate: int | None) -> int:
    """Return an effective centipawn value for a row, handling mate scores.

    Mate scores have no raw cp but represent a definitive advantage.
    We map them to ±_MATE_CP (White's perspective):
        mate > 0  → White mates → +_MATE_CP
        mate < 0  → Black mates → -_MATE_CP
    Normal cp values are returned as-is (clamping happens downstream).

    Args:
        cp:   Raw centipawn value from fishnet (White's perspective), or None.
        mate: Mate-in distance (positive = White mates, negative = Black mates),
              or None if not a mate position.

    Returns:
        Integer centipawn value (White's perspective).
    """
    if cp is not None:
        return int(cp)
    # Mate position: mate field holds the signed distance
    if mate is not None:
        return _MATE_CP if int(mate) > 0 else -_MATE_CP
    # Fallback (should not happen in well-formed data)
    return 0


def _cp_diff_scaled(cp_before: int, cp_after: int, white_to_move: bool) -> float:
    """Compute tanh-scaled cp diff from the mover's perspective.

    Args:
        cp_before: Centipawns before the move (White's perspective).
        cp_after:  Centipawns after the move (White's perspective).
        white_to_move: True if White played the move.

    Returns:
        Float in (-1, 1). Positive = good move for mover, negative = bad.
    """
    diff = cp_after - cp_before
    if not white_to_move:
        diff = -diff  # flip: improvement for Black = negative White delta
    diff = max(-_CP_CLAMP, min(_CP_CLAMP, diff))
    return math.tanh(diff / _CP_SCALE)


def _is_game_boundary(prev_fen: str, prev_move: str, next_fen: str) -> bool:
    """Detect game boundaries by comparing the first 4 FEN fields.

    Checks only piece placement, side to move, castling rights, and en-passant
    (fields 0-3) — skipping move counters which don't affect identity.
    Avoids a full python-chess round-trip; uses simple string comparison instead.
    Returns True if the rows belong to different games (or are corrupt).
    """
    # Quick prefix check: first two fields must change consistently.
    # Full board parse is avoided — just split and compare the position fields.
    p = prev_fen.split()
    n = next_fen.split()
    if len(p) < 4 or len(n) < 4:
        return True
    # Side to move must flip between consecutive rows
    expected_side = "b" if p[1] == "w" else "w"
    if n[1] != expected_side:
        return True
    return False


def stream_triples(
    limit: int,
    min_ply: int,
    max_ply: int,
    static_ratio: float = 0.01,
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Stream (fen, move_uci, cp_diff_scaled) triples from fishnet-evals.

    Game boundaries are detected by checking that side-to-move flips between
    consecutive rows (fast string check, no python-chess round-trip).

    Mate positions are included: their cp is mapped to ±_MATE_CP so the model
    learns that mating moves have maximum positive label.

    Static augmentation: for every real move record, a synthetic static record
    (move_uci=None, label=0.0) is interleaved at rate static_ratio.
    Both move and static records count toward `limit`.

    Yields dicts with keys: fen, move_uci (str or None), cp_diff_scaled.
    """
    from datasets import load_dataset

    log.info(
        "Worker %d streaming %s (limit=%d, ply %d-%d, static_ratio=%.2f)...",
        shard_id,
        _HF_REPO,
        limit,
        min_ply,
        max_ply,
        static_ratio,
    )
    ds = load_dataset(_HF_REPO, split="train", streaming=True)
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_id)

    # Deterministic interleaving: emit a static record every (1/static_ratio) move records.
    # Use a fractional accumulator to avoid clustering.
    static_accumulator = 0.0

    emitted = 0
    skipped_boundary = 0
    skipped_ply = 0
    emitted_static = 0
    total_seen = 0

    prev_row: dict | None = None
    ply_in_game: int = 0  # rough ply counter within current game

    for row in ds:
        if emitted >= limit:
            break

        total_seen += 1

        if prev_row is not None:
            # Fast game-boundary detection: side-to-move must flip on every row.
            # Avoids a full python-chess round-trip per record.
            if _is_game_boundary(prev_row["fen"], prev_row["move"], row["fen"]):
                skipped_boundary += 1
                prev_row = row
                ply_in_game = 0
                continue

            # Skip outside ply window
            if ply_in_game < min_ply or ply_in_game > max_ply:
                skipped_ply += 1
                prev_row = row
                ply_in_game += 1
                continue

            # Skip terminal rows (last position in a game — no move was played)
            if prev_row["move"] is None:
                prev_row = row
                ply_in_game += 1
                continue

            # Use effective cp: mate rows map to ±_MATE_CP instead of being skipped
            cp_before = _effective_cp(prev_row["cp"], prev_row.get("mate"))
            cp_after = _effective_cp(row["cp"], row.get("mate"))

            # FAST string split bypasses python-chess board parsing
            white_to_move = prev_row["fen"].split(" ")[1] == "w"

            diff = _cp_diff_scaled(cp_before, cp_after, white_to_move)

            yield {
                "fen": prev_row["fen"],
                "move_uci": prev_row["move"],
                "cp_diff_scaled": round(diff, 6),
            }
            emitted += 1

            # Interleave synthetic static record (before==after, label=0.0)
            if static_ratio > 0:
                static_accumulator += static_ratio
                if static_accumulator >= 1.0 and emitted < limit:
                    static_accumulator -= 1.0
                    yield {
                        "fen": prev_row["fen"],
                        "move_uci": None,
                        "cp_diff_scaled": 0.0,
                    }
                    emitted += 1
                    emitted_static += 1

            if emitted % 100_000 == 0:
                log.info(
                    "[Worker %d] Emitted %d / %d  (static=%d)  |  seen=%d  boundary=%d  ply=%d",
                    shard_id,
                    emitted,
                    limit,
                    emitted_static,
                    total_seen,
                    skipped_boundary,
                    skipped_ply,
                )

        prev_row = row
        ply_in_game += 1

    log.info(
        "[Worker %d] Done. Emitted=%d (static=%d)  seen=%d  boundary=%d  ply=%d",
        shard_id,
        emitted,
        emitted_static,
        total_seen,
        skipped_boundary,
        skipped_ply,
    )


def run_worker(args, shard_id, progress_counter=None):
    # Setup logger for the worker to suppress HTTP spam
    log.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    limit_per_worker = args.limit // args.workers + (
        1 if shard_id < args.limit % args.workers else 0
    )
    if limit_per_worker <= 0:
        return

    out_train = Path(f"{args.output}.{shard_id}")
    out_eval = Path(f"{args.eval_output}.{shard_id}")

    written_train = 0
    written_eval = 0
    last_update_count = 0

    with (
        out_train.open("w", encoding="utf-8") as f_train,
        out_eval.open("w", encoding="utf-8") as f_eval,
    ):
        for record in stream_triples(
            limit_per_worker, args.min_ply, args.max_ply, args.static_ratio, shard_id, args.workers
        ):
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if args.eval_ratio > 0 and random.random() < args.eval_ratio:
                f_eval.write(line)
                written_eval += 1
            else:
                f_train.write(line)
                written_train += 1
            
            # Update shared counter every 10,000 records to reduce lock contention
            if progress_counter and (written_train + written_eval) - last_update_count >= 10000:
                with progress_counter.get_lock():
                    progress_counter.value += (written_train + written_eval) - last_update_count
                last_update_count = (written_train + written_eval)
        
        # Final update
        if progress_counter:
            with progress_counter.get_lock():
                progress_counter.value += (written_train + written_eval) - last_update_count

    log.info("[Worker %d] Saved %d train records to %s", shard_id, written_train, out_train)
    log.info("[Worker %d] Saved %d eval records to %s", shard_id, written_eval, out_eval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/processed/encoder_pretrain.jsonl",
        help="Output JSONL path for training data",
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed/encoder_pretrain_eval.jsonl",
        help="Output JSONL path for evaluation data",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.01,
        help="Fraction of records to write to the evaluation file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000_000,
        help="Max number of triples to emit",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=4,
        help="Skip positions before this ply in the game (avoids book moves)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=80,
        help="Skip positions after this ply (avoids drawn-out endgames)",
    )
    parser.add_argument(
        "--static-ratio",
        type=float,
        default=0.01,
        help=(
            "Fraction of move records to pair with a synthetic static record "
            "(move_uci=null, label=0.0). 0.01 = one static per 100 move records. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of multi-processing workers to speed up dataset processing.",
    )
    args = parser.parse_args()

    # Silence main process HTTP requests
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    out_train = Path(args.output)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_eval = Path(args.eval_output)
    out_eval.parent.mkdir(parents=True, exist_ok=True)

    if args.workers == 1:
        run_worker(args, 0)
        import os

        os.rename(f"{args.output}.0", args.output)
        os.rename(f"{args.eval_output}.0", args.eval_output)
    else:
        import multiprocessing as mp
        import os
        import shutil
        import time
        from tqdm import tqdm

        progress_counter = mp.Value("i", 0)
        processes = []
        for i in range(args.workers):
            p = mp.Process(target=run_worker, args=(args, i, progress_counter))
            p.start()
            processes.append(p)

        # Parent process manages the tqdm bar
        with tqdm(total=args.limit, desc="Generating 1B positions") as pbar:
            last_val = 0
            while any(p.is_alive() for p in processes):
                current_val = progress_counter.value
                pbar.update(current_val - last_val)
                last_val = current_val
                time.sleep(1.0)
            
            # Final update
            current_val = progress_counter.value
            pbar.update(current_val - last_val)

        for p in processes:
            p.join()

        log.info("Concatenating worker outputs...")
        with out_train.open("wb") as outfile:
            for i in range(args.workers):
                worker_file = f"{args.output}.{i}"
                if os.path.exists(worker_file):
                    with open(worker_file, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(worker_file)

        with out_eval.open("wb") as outfile:
            for i in range(args.workers):
                worker_file = f"{args.eval_output}.{i}"
                if os.path.exists(worker_file):
                    with open(worker_file, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
                    os.remove(worker_file)

        log.info("Finished concatenating to %s and %s", args.output, args.eval_output)


if __name__ == "__main__":
    main()
