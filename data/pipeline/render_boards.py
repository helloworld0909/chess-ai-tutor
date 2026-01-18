"""Render board images from positions for VL model training.

Generates PNG images from FEN strings using python-chess and cairosvg.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import chess
import chess.svg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_position(
    fen: str,
    output_path: Path,
    size: int = 400,
    last_move: str | None = None,
    flip: bool = False,
) -> bool:
    """Render a single position to PNG.

    Args:
        fen: Position in FEN notation
        output_path: Path to save PNG
        size: Image size in pixels
        last_move: Last move to highlight (UCI notation)
        flip: Show from Black's perspective

    Returns:
        True if successful
    """
    try:
        import cairosvg
    except ImportError:
        logger.error("cairosvg is required. Install with: pip install cairosvg")
        return False

    try:
        board = chess.Board(fen)

        # Parse last move if provided
        last_move_obj = None
        if last_move:
            try:
                last_move_obj = chess.Move.from_uci(last_move)
            except ValueError:
                pass

        # Generate SVG
        svg = chess.svg.board(
            board,
            size=size,
            lastmove=last_move_obj,
            flipped=flip,
        )

        # Convert to PNG
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(output_path))

        return True

    except Exception as e:
        logger.warning(f"Error rendering {fen[:30]}...: {e}")
        return False


def render_position_worker(args: tuple) -> tuple[str, bool]:
    """Worker function for parallel rendering.

    Args:
        args: Tuple of (position_id, fen, output_path, size, last_move, flip)

    Returns:
        Tuple of (position_id, success)
    """
    position_id, fen, output_path, size, last_move, flip = args

    success = render_position(
        fen=fen,
        output_path=Path(output_path),
        size=size,
        last_move=last_move,
        flip=flip,
    )

    return position_id, success


def load_positions(input_path: Path) -> Iterator[dict]:
    """Load positions from JSONL file.

    Args:
        input_path: Path to JSONL file

    Yields:
        Position dictionaries
    """
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def render_from_jsonl(
    input_path: Path,
    output_dir: Path,
    size: int = 400,
    max_workers: int = 4,
    max_positions: int | None = None,
    highlight_moves: bool = True,
) -> int:
    """Render all positions from a JSONL file.

    Args:
        input_path: Path to JSONL with positions
        output_dir: Directory to save images
        size: Image size in pixels
        max_workers: Number of parallel workers
        max_positions: Maximum positions to render
        highlight_moves: Highlight the played move

    Returns:
        Number of images rendered
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect render tasks
    tasks = []
    for i, pos in enumerate(load_positions(input_path)):
        if max_positions and i >= max_positions:
            break

        fen = pos.get("fen", "")
        if not fen:
            continue

        # Generate unique ID for the position
        game_id = pos.get("game_id", f"pos_{i}")
        move_num = pos.get("move_number", i)
        position_id = f"{game_id}_{move_num}"

        output_path = output_dir / f"{position_id}.png"

        # Skip if already rendered
        if output_path.exists():
            continue

        last_move = pos.get("move_uci") if highlight_moves else None

        # Determine perspective (show from moving side's view)
        side = pos.get("side_to_move", "white")
        flip = side == "black"

        tasks.append((
            position_id,
            fen,
            str(output_path),
            size,
            last_move,
            flip,
        ))

    if not tasks:
        logger.info("No new positions to render")
        return 0

    logger.info(f"Rendering {len(tasks)} board images...")

    # Render in parallel
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_position_worker, task) for task in tasks]

        for i, future in enumerate(as_completed(futures)):
            position_id, success = future.result()
            if success:
                success_count += 1

            if (i + 1) % 500 == 0:
                logger.info(f"Rendered {i + 1}/{len(tasks)} images...")

    logger.info(f"Successfully rendered {success_count} images")
    return success_count


def create_image_manifest(
    image_dir: Path,
    positions_path: Path,
    output_path: Path,
) -> int:
    """Create a manifest mapping positions to images.

    Args:
        image_dir: Directory with rendered images
        positions_path: JSONL with position data
        output_path: Path for manifest JSONL

    Returns:
        Number of entries in manifest
    """
    # Get available images
    available_images = {p.stem: p for p in image_dir.glob("*.png")}
    logger.info(f"Found {len(available_images)} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, pos in enumerate(load_positions(positions_path)):
            game_id = pos.get("game_id", f"pos_{i}")
            move_num = pos.get("move_number", i)
            position_id = f"{game_id}_{move_num}"

            if position_id in available_images:
                manifest_entry = {
                    "position_id": position_id,
                    "image_path": str(available_images[position_id]),
                    "fen": pos.get("fen", ""),
                    "move_uci": pos.get("move_uci", ""),
                    "move_san": pos.get("move_san", ""),
                    "annotation": pos.get("annotation", ""),
                    "concepts": pos.get("concepts", []),
                }
                f_out.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Created manifest with {count} entries")
    return count


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Render board images for VL training")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/processed/augmented.jsonl"),
        help="Input JSONL with positions",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/images"),
        help="Output directory for images",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=448,
        help="Image size in pixels",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum positions to render",
    )
    parser.add_argument(
        "--no-highlight",
        action="store_true",
        help="Don't highlight the played move",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Create image manifest at this path",
    )

    args = parser.parse_args()

    # Render images
    count = render_from_jsonl(
        args.input,
        args.output,
        size=args.size,
        max_workers=args.workers,
        max_positions=args.max,
        highlight_moves=not args.no_highlight,
    )

    print(f"Rendered {count} images to {args.output}")

    # Create manifest if requested
    if args.manifest:
        manifest_count = create_image_manifest(
            args.output,
            args.input,
            args.manifest,
        )
        print(f"Created manifest with {manifest_count} entries at {args.manifest}")


if __name__ == "__main__":
    main()
