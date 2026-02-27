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
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ELO tiers for stratified sampling
ELO_TIERS = {
    "amateur": (1000, 1600),
    "intermediate": (1600, 2000),
    "strong": (2000, 9999),
}

# Target fraction per tier
TIER_FRACTIONS = {"amateur": 0.50, "intermediate": 0.35, "strong": 0.15}

# Skip early opening moves — too theory-dependent
MIN_MOVE_NUMBER = 8

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


def iter_lichess_games(
    target_positions: int,
    tier_fractions: dict[str, float] = TIER_FRACTIONS,
    seed: int = 42,
) -> list[dict]:
    """Stream games from austindavis/lichess_uci and return sampled positions.

    Returns a list of dicts: {fen, move_san, move_uci, white_elo, black_elo, tier}.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: uv add datasets")

    rng = random.Random(seed)
    tier_targets = {t: int(target_positions * f) for t, f in tier_fractions.items()}
    tier_counts = {t: 0 for t in tier_fractions}
    positions: list[dict] = []

    logger.info("Loading austindavis/lichess_uci (streaming)...")
    # Use a single monthly split to avoid downloading the full 5B-game dataset
    ds = load_dataset(
        "austindavis/lichess_uci",
        "201801-headers",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    games_seen = 0
    for game in ds:
        if all(tier_counts[t] >= tier_targets[t] for t in tier_fractions):
            break

        games_seen += 1
        white_elo = int(game.get("whiteelo") or 0)
        black_elo = int(game.get("blackelo") or 0)
        avg_elo = (white_elo + black_elo) // 2 if white_elo and black_elo else 0
        if avg_elo == 0:
            continue

        tier = next((t for t, (lo, hi) in ELO_TIERS.items() if lo <= avg_elo < hi), None)
        if tier is None or tier_counts[tier] >= tier_targets[tier]:
            continue

        # Parse moves from transcript field
        transcript = game.get("transcript", "")
        if not transcript:
            continue

        extracted = _extract_positions_from_transcript(transcript, white_elo, black_elo, tier, rng)
        if extracted:
            positions.append(extracted)
            tier_counts[tier] += 1

        if games_seen % 10000 == 0:
            logger.info(
                "Games: %d | Positions: %s",
                games_seen,
                " | ".join(f"{t}={tier_counts[t]}" for t in tier_fractions),
            )

    logger.info(
        "Sampled %d positions from %d games. Tier breakdown: %s",
        len(positions),
        games_seen,
        tier_counts,
    )
    return positions


def _extract_positions_from_transcript(
    transcript: str,
    white_elo: int,
    black_elo: int,
    tier: str,
    rng: random.Random,
) -> dict | None:
    """Replay a UCI transcript and pick one interesting position to sample."""
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
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "tier": tier,
                }
            )
        except Exception:
            return None

    if not candidate_positions:
        return None

    # Pick one position at random from middle-game range
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
                        "white_elo": pos["white_elo"],
                        "black_elo": pos["black_elo"],
                        "elo_tier": pos["tier"],
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

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        async for record in process_batch(positions, args.stockfish, args.workers):
            f.write(json.dumps(record) + "\n")
            written += 1
            if written % 1000 == 0:
                logger.info("Written %d / %d (skipped %d)", written, len(positions), skipped)
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
