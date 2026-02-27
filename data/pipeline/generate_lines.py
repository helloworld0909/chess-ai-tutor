"""Generate Stockfish key-line data from Lichess games for Stage 1 SFT/GRPO.

For each sampled (FEN, move) position, runs Stockfish multipv=3 to get key lines,
maps centipawn scores to eval labels (no raw numbers in output), and emits JSONL.

Output format per record:
    {
      "fen": "<FEN before the move>",
      "move_san": "<move played in SAN>",
      "move_uci": "<move played in UCI>",
      "lines": [
        "LINE 1: e4 → Nf6 → Nc3 | eval: equal",
        "LINE 2: e4 → d5 → exd5 | eval: good for white",
        "LINE 3: e4 → c5 → Nf3 | eval: very good for white"
      ],
      "metadata": {
        "source": "lichess_lines",
        "white_elo": 1523,
        "black_elo": 1487,
        "elo_tier": "amateur"
      }
    }

Usage:
    uv run python data/pipeline/generate_lines.py \\
        --stockfish /home/zheng/.local/bin/stockfish \\
        --output data/processed/lines.jsonl \\
        --target 50000 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import AsyncIterator

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from chess_mcp.stockfish import Stockfish

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# Suppress noisy HTTP logs from httpx (used by huggingface_hub during dataset download)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Skip the first N full moves (each full move = 2 ply: one white, one black)
# 2 → skip only the first 2 full moves (4 ply), sample from move 3 onward
MIN_MOVE_NUMBER = 2

# Max ply per game to sample from (avoid dead endgames)
MAX_PLY = 80

# Stockfish settings
ANALYSIS_DEPTH = 15
MULTIPV = 3

# Centipawn → eval label mapping (always from white's perspective)
CP_BANDS = [
    (300, "winning for white"),
    (101, "good for white"),
    (-100, "equal"),
    (-300, "good for black"),
]
LABEL_WINNING_BLACK = "winning for black"


def cp_to_label(cp: int | None, is_mate: bool = False, mate_value: int = 0) -> str:
    """Map a centipawn score (white perspective) to an eval label."""
    if is_mate:
        return "winning for white" if mate_value > 0 else "winning for black"
    if cp is None:
        return "equal"
    for threshold, label in CP_BANDS:
        if cp >= threshold:
            return label
    return LABEL_WINNING_BLACK


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


_HF_REPO = "austindavis/lichess_uci"
# data/ shards contain {site, transcript} for all games in that month.
# Each shard is ~150 MB and holds ~1.3M games. We download one at a time.
_HF_SHARDS = [f"data/201801-{i:05d}-of-00013.parquet" for i in range(13)]


def iter_lichess_games(
    target_positions: int,
    seed: int = 42,
) -> list[dict]:
    """Sample positions from austindavis/lichess_uci parquet shards.

    Returns a list of dicts: {fen, move_san, move_uci}.
    Downloads one ~150 MB shard at a time (cached to HF hub after first run).
    Stops as soon as target_positions is reached — no need to load all rows.
    """
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Run: uv add pyarrow huggingface_hub")

    rng = random.Random(seed)
    positions: list[dict] = []
    games_seen = 0

    for shard in _HF_SHARDS:
        if len(positions) >= target_positions:
            break
        logger.info("Loading shard %s...", shard)
        local_path = hf_hub_download(repo_id=_HF_REPO, filename=shard, repo_type="dataset")
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=10000, columns=["transcript"]):
            if len(positions) >= target_positions:
                break
            for transcript in batch.column("transcript").to_pylist():
                if len(positions) >= target_positions:
                    break
                if not transcript:
                    continue
                games_seen += 1
                extracted = _extract_positions_from_transcript(transcript, rng)
                if extracted:
                    positions.append(extracted)
                if games_seen % 10000 == 0:
                    logger.info(
                        "Games scanned: %d | Positions collected: %d", games_seen, len(positions)
                    )

    logger.info("Sampled %d positions from %d games.", len(positions), games_seen)
    return positions


def _extract_positions_from_transcript(
    transcript: str,
    rng: random.Random,
) -> dict | None:
    """Replay a UCI transcript and pick one random position to sample."""
    # transcript is space-separated UCI moves: "e2e4 e7e5 g1f3 ..."
    uci_moves = transcript.strip().split()
    if len(uci_moves) < MIN_MOVE_NUMBER * 2:
        return None

    board = chess.Board()
    candidate_positions: list[dict] = []

    for ply, uci in enumerate(uci_moves):
        if ply < MIN_MOVE_NUMBER * 2:
            try:
                board.push_uci(uci)
            except Exception:
                return None
            continue
        if ply > MAX_PLY:
            break

        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return None
            fen_before = board.fen()
            san = board.san(move)
            board.push(move)
            candidate_positions.append(
                {
                    "fen": fen_before,
                    "move_san": san,
                    "move_uci": uci,
                }
            )
        except Exception:
            return None

    if not candidate_positions:
        return None

    return rng.choice(candidate_positions)


# ---------------------------------------------------------------------------
# Move annotation
# ---------------------------------------------------------------------------

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def annotate_move(board: chess.Board, move: chess.Move) -> str:
    """Return a simple deterministic annotation for a move.

    Covers castling, promotion, captures, and checks. This placeholder
    gives the output format its shape; richer purpose annotations come
    later via GRPO/Haiku fine-tuning.
    """
    if board.is_castling(move):
        side = "kingside" if board.is_kingside_castling(move) else "queenside"
        suffix = " check" if board.gives_check(move) else ""
        return f"castle {side}{suffix}"

    piece = board.piece_at(move.from_square)
    piece_name = _PIECE_NAMES.get(piece.piece_type, "piece") if piece else "piece"

    if move.promotion:
        promo_name = _PIECE_NAMES.get(move.promotion, "queen")
        suffix = " check" if board.gives_check(move) else ""
        return f"promote to {promo_name}{suffix}"

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            cap_name = _PIECE_NAMES.get(captured.piece_type, "piece")
            suffix = " check" if board.gives_check(move) else ""
            return f"capture {cap_name}{suffix}"
        # en passant — captured pawn is not on to_square
        suffix = " check" if board.gives_check(move) else ""
        return f"capture pawn{suffix}"

    suffix = " check" if board.gives_check(move) else ""
    return f"move {piece_name}{suffix}"


def score_annotation_structural(board: chess.Board, move: chess.Move, annotation: str) -> float:
    """Score a model-generated annotation against ground-truth structural facts.

    Checks three binary facts derivable from python-chess at zero cost:
    - capture vs non-capture
    - correct piece name mentioned
    - check flag matches

    Returns +1.0 if all facts are correct, -1.0 if any is wrong.
    Used as R4a in the GRPO reward during training.
    """
    ann = annotation.lower()

    is_capture = board.is_capture(move)
    mentions_capture = "capture" in ann
    if is_capture != mentions_capture:
        return -1.0

    if is_capture:
        # For captures, annotation names the captured piece (not the mover)
        captured = board.piece_at(move.to_square)
        if captured is not None:
            cap_name = _PIECE_NAMES.get(captured.piece_type, "piece")
            if cap_name not in ann:
                return -1.0
    else:
        # For non-captures, annotation names the moving piece
        piece = board.piece_at(move.from_square)
        if piece is not None:
            piece_name = _PIECE_NAMES.get(piece.piece_type, "piece")
            if piece_name not in ann:
                return -1.0

    gives_check = board.gives_check(move)
    mentions_check = "check" in ann
    if gives_check != mentions_check:
        return -1.0

    return 1.0


# ---------------------------------------------------------------------------
# Line generation
# ---------------------------------------------------------------------------


async def generate_lines_for_position(
    sf: Stockfish,
    fen: str,
    move_uci: str,
    move_san: str,
) -> list[str]:
    """Run Stockfish multipv from position after move_uci, return formatted lines."""
    # Play the move to get the position the lines start from
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
    except Exception as e:
        logger.debug("Invalid move %s in position %s: %s", move_uci, fen, e)
        return []

    post_move_fen = board.fen()
    analysis = await sf.analyze(post_move_fen, depth=ANALYSIS_DEPTH, multipv=MULTIPV)

    formatted_lines = []
    for i, line in enumerate(analysis.lines, 1):
        if not line.pv:
            continue

        # Convert UCI PV to SAN with annotations, playing out from post-move position
        line_board = chess.Board(post_move_fen)
        annotated_moves: list[str] = []
        valid = True
        for uci_mv in line.pv:
            try:
                mv = chess.Move.from_uci(uci_mv)
                san = line_board.san(mv)
                annotation = annotate_move(line_board, mv)
                annotated_moves.append(f"{san} ({annotation})")
                line_board.push(mv)
            except Exception:
                valid = False
                break
        if not valid or not annotated_moves:
            continue

        # Eval label from final position score (white perspective)
        score = line.score
        label = cp_to_label(
            cp=score.centipawns,
            is_mate=(score.mate_in is not None),
            mate_value=score.mate_in or 0,
        )

        moves_str = " → ".join(annotated_moves)
        formatted_lines.append(f"LINE {i}: {moves_str} | eval: {label}")

    return formatted_lines


async def process_batch(
    positions: list[dict],
    stockfish_path: str,
    workers: int,
) -> AsyncIterator[dict]:
    """Process positions in parallel with a pool of Stockfish workers."""
    semaphore = asyncio.Semaphore(workers)
    sf_instances = [
        Stockfish(path=stockfish_path, depth=ANALYSIS_DEPTH, threads=1, hash_mb=64)
        for _ in range(workers)
    ]
    # Start all instances
    for sf in sf_instances:
        await sf.start()

    sf_queue: asyncio.Queue = asyncio.Queue()
    for sf in sf_instances:
        await sf_queue.put(sf)

    async def process_one(pos: dict) -> dict | None:
        async with semaphore:
            sf = await sf_queue.get()
            try:
                lines = await generate_lines_for_position(
                    sf, pos["fen"], pos["move_uci"], pos["move_san"]
                )
                if not lines:
                    return None
                return {
                    "fen": pos["fen"],
                    "move_san": pos["move_san"],
                    "move_uci": pos["move_uci"],
                    "lines": lines,
                    "metadata": {
                        "source": "lichess_lines",
                    },
                }
            except Exception as e:
                logger.debug("Error processing position %s: %s", pos["fen"], e)
                return None
            finally:
                await sf_queue.put(sf)

    tasks = [process_one(pos) for pos in positions]
    results = await asyncio.gather(*tasks)

    for sf in sf_instances:
        await sf.stop()

    for result in results:
        if result is not None:
            yield result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling %d positions from Lichess...", args.target)
    positions = iter_lichess_games(target_positions=args.target, seed=args.seed)
    logger.info("Got %d positions. Running Stockfish analysis...", len(positions))

    # Process in chunks of 1000 so results are flushed to disk progressively
    # rather than holding all 30k futures in memory at once.
    CHUNK = 1000
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk_start in range(0, len(positions), CHUNK):
            chunk = positions[chunk_start : chunk_start + CHUNK]
            async for record in process_batch(chunk, args.stockfish, args.workers):
                f.write(json.dumps(record) + "\n")
                written += 1
            f.flush()
            logger.info(
                "Progress: %d / %d written (chunk %d/%d)",
                written,
                len(positions),
                chunk_start // CHUNK + 1,
                (len(positions) + CHUNK - 1) // CHUNK,
            )

    skipped = len(positions) - written
    logger.info("Done. Written: %d | Skipped: %d | Output: %s", written, skipped, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stockfish key-line data from Lichess")
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (default: $STOCKFISH_PATH or 'stockfish')",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lines.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50000,
        help="Target number of positions to sample",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel Stockfish workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for position sampling",
    )
    args = parser.parse_args()

    if args.stockfish is None:
        import os

        args.stockfish = os.environ.get("STOCKFISH_PATH", "stockfish")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
