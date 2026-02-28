"""Augment extracted annotations with Stockfish evaluations.

Adds engine analysis to validate move quality without including
raw scores in training output.
"""

from __future__ import annotations

import asyncio
import json
import logging

# Add parent to path for imports
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import chess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chess_mcp.stockfish import Stockfish

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentedPosition:
    """Position with both human annotation and engine data."""

    # Original annotation data
    fen: str
    move_uci: str
    move_san: str
    annotation: str
    source_file: str
    game_id: str
    move_number: int
    side_to_move: str
    concepts: list[str] | None

    # Engine augmentation (stored separately, not in training output)
    best_move: str
    is_best_move: bool
    centipawn_loss: int
    engine_classification: str
    win_probability: float
    refutation_line: list[str]


def classify_by_cp_loss(cp_loss: int) -> str:
    """Classify move quality by centipawn loss."""
    if cp_loss <= 10:
        return "Best"
    elif cp_loss <= 30:
        return "Great"
    elif cp_loss <= 80:
        return "Good"
    elif cp_loss <= 150:
        return "Inaccuracy"
    elif cp_loss <= 300:
        return "Mistake"
    else:
        return "Blunder"


async def augment_position(
    position: dict,
    stockfish: Stockfish,
    depth: int = 20,
) -> dict | None:
    """Augment a single position with engine analysis.

    Args:
        position: Position dictionary from annotations.jsonl
        stockfish: Stockfish instance
        depth: Analysis depth

    Returns:
        Augmented position dictionary or None if error
    """
    fen = position["fen"]
    move = position["move_uci"]

    try:
        # Validate position
        board = chess.Board(fen)
        chess_move = chess.Move.from_uci(move)

        if chess_move not in board.legal_moves:
            logger.warning(f"Illegal move {move} in position")
            return None

        # Get engine comparison
        comparison = await stockfish.compare_moves(fen, move, depth)

        if "error" in comparison:
            logger.warning(f"Engine error: {comparison['error']}")
            return None

        cp_loss = comparison.get("cp_loss", 0)
        best_move = comparison.get("best_move", "")
        is_best = comparison.get("is_best", False)
        pv = comparison.get("pv", [])

        # Get win probability
        analysis = await stockfish.analyze(fen, depth=depth)
        win_prob = analysis.score.win_probability

        # Create augmented data
        augmented = {
            **position,
            "engine_data": {
                "best_move": best_move,
                "is_best_move": is_best,
                "centipawn_loss": cp_loss,
                "engine_classification": classify_by_cp_loss(cp_loss),
                "win_probability": round(win_prob, 3),
                "refutation_line": pv[:5],
            },
        }

        return augmented

    except Exception as e:
        logger.warning(f"Error augmenting position: {e}")
        return None


async def augment_batch(
    positions: list[dict],
    stockfish: Stockfish,
    depth: int = 20,
) -> list[dict]:
    """Augment a batch of positions.

    Args:
        positions: List of position dictionaries
        stockfish: Stockfish instance
        depth: Analysis depth

    Returns:
        List of augmented positions
    """
    results = []

    for pos in positions:
        augmented = await augment_position(pos, stockfish, depth)
        if augmented:
            results.append(augmented)

    return results


async def augment_file(
    input_path: Path,
    output_path: Path,
    stockfish_path: str | None = None,
    depth: int = 20,
    batch_size: int = 100,
    max_positions: int | None = None,
) -> int:
    """Augment all positions in a JSONL file.

    Args:
        input_path: Input JSONL with annotations
        output_path: Output JSONL with augmented data
        stockfish_path: Path to Stockfish binary
        depth: Analysis depth
        batch_size: Positions to process before writing
        max_positions: Maximum positions to process (for testing)

    Returns:
        Number of positions augmented
    """
    # Initialize Stockfish
    stockfish = Stockfish(
        path=stockfish_path,
        depth=depth,
        threads=4,
        hash_mb=512,
    )

    await stockfish.start()
    logger.info("Stockfish initialized")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        batch = []

        with open(input_path, "r", encoding="utf-8") as f_in:
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if max_positions and count >= max_positions:
                        break

                    try:
                        position = json.loads(line)
                        batch.append(position)

                        if len(batch) >= batch_size:
                            augmented = await augment_batch(batch, stockfish, depth)
                            for aug in augmented:
                                f_out.write(json.dumps(aug, ensure_ascii=False) + "\n")
                            count += len(augmented)
                            batch = []

                            if count % 500 == 0:
                                logger.info(f"Augmented {count} positions...")

                    except json.JSONDecodeError:
                        continue

                # Process remaining batch
                if batch:
                    augmented = await augment_batch(batch, stockfish, depth)
                    for aug in augmented:
                        f_out.write(json.dumps(aug, ensure_ascii=False) + "\n")
                    count += len(augmented)

        logger.info(f"Total: {count} positions augmented")
        return count

    finally:
        await stockfish.stop()


def create_training_sample(augmented: dict, include_image_path: bool = False) -> dict:
    """Convert augmented position to training format.

    Creates the conversation format for fine-tuning,
    WITHOUT including engine scores in the output.

    Args:
        augmented: Augmented position dictionary
        include_image_path: Include image path for VL training

    Returns:
        Training sample in messages format
    """
    fen = augmented["fen"]
    move_san = augmented["move_san"]
    annotation = augmented["annotation"]
    concepts = augmented.get("concepts", [])
    engine_data = augmented.get("engine_data", {})

    # System prompt
    system_content = (
        "You are a chess instructor teaching a student. "
        "Explain positions clearly using chess concepts, not engine scores. "
        "Think through positions step by step."
    )

    # User message
    user_text = f"I played {move_san} here. What do you think of this move?"

    if include_image_path:
        user_content = [
            {
                "type": "image",
                "image": f"images/{augmented['game_id']}_{augmented['move_number']}.png",
            },
            {"type": "text", "text": user_text},
        ]
    else:
        # Include ASCII board for text-only training
        import chess

        board = chess.Board(fen)
        ascii_board = str(board)
        user_content = f"Position:\n```\n{ascii_board}\n```\nFEN: {fen}\n\n{user_text}"

    # Assistant response - use the human annotation
    # We don't include engine classification, just the pedagogical explanation
    engine_class = engine_data.get("engine_classification", "")
    is_best = engine_data.get("is_best_move", False)

    # Build response with thinking
    thinking = f"Let me analyze the move {move_san}. "

    if concepts:
        thinking += f"This position involves {', '.join(concepts)}. "

    thinking += "I'll explain the strategic and tactical aspects."

    # The actual response uses the human annotation
    response = f"<think>{thinking}</think>\n\n{annotation}"

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ],
        "fen": fen,
        "move_played": augmented["move_uci"],
        "source": augmented.get("source_file", ""),
        "concepts": concepts,
        # Engine data stored separately for verification
        "_engine_data": engine_data,
    }


async def create_training_dataset(
    augmented_path: Path,
    output_path: Path,
    include_images: bool = False,
) -> int:
    """Create training dataset from augmented positions.

    Args:
        augmented_path: Path to augmented JSONL
        output_path: Path for training JSONL
        include_images: Include image paths for VL training

    Returns:
        Number of samples created
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(augmented_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    augmented = json.loads(line)
                    sample = create_training_sample(augmented, include_images)
                    f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    continue

    logger.info(f"Created {count} training samples")
    return count


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Augment annotations with Stockfish")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/annotations.jsonl"),
        help="Input JSONL with annotations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/augmented.jsonl"),
        help="Output JSONL with engine data",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Analysis depth",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum positions to process",
    )
    parser.add_argument(
        "--create-training",
        action="store_true",
        help="Also create training dataset",
    )

    args = parser.parse_args()

    # Run augmentation
    count = asyncio.run(
        augment_file(
            args.input,
            args.output,
            stockfish_path=args.stockfish,
            depth=args.depth,
            max_positions=args.max,
        )
    )

    print(f"Augmented {count} positions to {args.output}")

    # Create training dataset if requested
    if args.create_training:
        training_path = args.output.parent / "train.jsonl"
        train_count = asyncio.run(create_training_dataset(args.output, training_path))
        print(f"Created {train_count} training samples at {training_path}")


if __name__ == "__main__":
    main()
