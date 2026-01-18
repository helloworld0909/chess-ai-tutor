"""Extract annotated positions from PGN files.

Parses PGN games and extracts positions with meaningful annotations
for fine-tuning data generation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnnotatedPosition:
    """A chess position with annotation."""

    fen: str
    move_uci: str
    move_san: str
    annotation: str
    nag: list[int]  # Numeric Annotation Glyphs
    source_file: str
    game_id: str
    move_number: int
    side_to_move: str
    opening: str | None = None
    white_player: str | None = None
    black_player: str | None = None
    event: str | None = None
    concepts: list[str] | None = None


# Minimum annotation length to be considered meaningful
MIN_ANNOTATION_LENGTH = 20

# Keywords that indicate instructive content
INSTRUCTIVE_KEYWORDS = [
    # Explanatory
    "because", "since", "therefore", "thus", "hence",
    "in order to", "so that", "allowing",
    # Chess concepts
    "weak", "strong", "control", "pressure",
    "pawn structure", "outpost", "file", "diagonal",
    "attack", "defend", "threat", "pin", "fork",
    "discovery", "skewer", "sacrifice", "exchange",
    "development", "center", "initiative", "tempo",
    "king safety", "castle", "activity",
    # Strategic
    "plan", "idea", "strategy", "positional",
    "advantage", "compensation", "equality",
    # Evaluation
    "better", "worse", "winning", "losing",
    "unclear", "complicated", "sharp",
]


def is_quality_annotation(comment: str) -> bool:
    """Check if an annotation is instructive enough for training.

    Args:
        comment: The annotation text

    Returns:
        True if annotation meets quality criteria
    """
    if not comment or len(comment) < MIN_ANNOTATION_LENGTH:
        return False

    # Skip pure engine output
    if comment.startswith("+") or comment.startswith("-"):
        if re.match(r"^[+-]?\d+\.\d+", comment):
            return False

    # Skip move-only comments like "1.e4 e5 2.Nf3"
    if re.match(r"^\d+\.\s*[A-Za-z]", comment):
        move_pattern = r"\d+\.\s*[A-Za-z][a-h1-8x+#=]*"
        if len(re.findall(move_pattern, comment)) > len(comment.split()) * 0.5:
            return False

    # Check for instructive keywords
    comment_lower = comment.lower()
    return any(keyword in comment_lower for keyword in INSTRUCTIVE_KEYWORDS)


def extract_concepts(comment: str) -> list[str]:
    """Extract chess concepts mentioned in an annotation.

    Args:
        comment: The annotation text

    Returns:
        List of identified concepts
    """
    concepts = []
    comment_lower = comment.lower()

    concept_patterns = {
        "pawn structure": ["pawn structure", "isolated pawn", "doubled pawn",
                          "backward pawn", "passed pawn", "pawn chain",
                          "hanging pawns", "pawn majority"],
        "piece activity": ["active", "passive", "piece activity", "coordination",
                          "good bishop", "bad bishop", "knight outpost"],
        "king safety": ["king safety", "castle", "king exposed", "shelter",
                       "attack on king", "king in center"],
        "center control": ["center", "central", "d4", "e4", "d5", "e5",
                          "control of center"],
        "tactics": ["pin", "fork", "skewer", "discovery", "zwischenzug",
                   "in-between", "double attack", "remove defender",
                   "overloaded", "deflection"],
        "development": ["development", "develop", "piece out", "undeveloped",
                       "lead in development"],
        "space": ["space", "cramped", "space advantage"],
        "initiative": ["initiative", "tempo", "time", "attacking"],
        "endgame": ["endgame", "ending", "king activity", "opposition",
                   "outside passed pawn", "rook ending"],
        "opening": ["opening", "theory", "book move", "novelty"],
    }

    for concept, keywords in concept_patterns.items():
        if any(kw in comment_lower for kw in keywords):
            concepts.append(concept)

    return concepts


def nag_to_text(nag: int) -> str | None:
    """Convert Numeric Annotation Glyph to text.

    Args:
        nag: NAG code

    Returns:
        Text description or None
    """
    nag_meanings = {
        1: "good move",
        2: "poor move",
        3: "very good move",
        4: "very poor move",
        5: "speculative move",
        6: "questionable move",
        7: "forced move",
        10: "drawish position",
        13: "unclear position",
        14: "slight advantage for White",
        15: "slight advantage for Black",
        16: "moderate advantage for White",
        17: "moderate advantage for Black",
        18: "decisive advantage for White",
        19: "decisive advantage for Black",
        22: "zugzwang",
        32: "development advantage",
        36: "initiative",
        40: "attack",
        132: "counterplay",
        140: "with the idea",
        146: "novelty",
    }
    return nag_meanings.get(nag)


def extract_from_game(
    game: chess.pgn.Game,
    source_file: str,
) -> Iterator[AnnotatedPosition]:
    """Extract annotated positions from a single game.

    Args:
        game: Parsed PGN game
        source_file: Path to source file

    Yields:
        AnnotatedPosition for each meaningful annotation
    """
    # Get game metadata
    headers = game.headers
    white = headers.get("White", "Unknown")
    black = headers.get("Black", "Unknown")
    event = headers.get("Event", "")
    opening = headers.get("Opening", headers.get("ECO", ""))
    game_id = f"{white}_{black}_{headers.get('Date', '')}".replace(" ", "_")

    board = game.board()
    node = game

    move_number = 0

    while node.variations:
        next_node = node.variation(0)

        if next_node.move:
            # Get annotation
            comment = next_node.comment.strip() if next_node.comment else ""
            nags = list(next_node.nags)

            # Check if annotation is quality
            if is_quality_annotation(comment) or nags:
                # Get position before the move
                fen = board.fen()
                move = next_node.move
                move_san = board.san(move)
                side = "white" if board.turn == chess.WHITE else "black"

                # Combine NAG meanings with comment
                full_annotation = comment
                for nag in nags:
                    nag_text = nag_to_text(nag)
                    if nag_text and nag_text not in full_annotation.lower():
                        full_annotation = f"[{nag_text}] {full_annotation}"

                if is_quality_annotation(full_annotation):
                    concepts = extract_concepts(full_annotation)

                    yield AnnotatedPosition(
                        fen=fen,
                        move_uci=move.uci(),
                        move_san=move_san,
                        annotation=full_annotation,
                        nag=nags,
                        source_file=source_file,
                        game_id=game_id,
                        move_number=move_number,
                        side_to_move=side,
                        opening=opening if opening else None,
                        white_player=white,
                        black_player=black,
                        event=event if event else None,
                        concepts=concepts if concepts else None,
                    )

            # Apply move
            board.push(next_node.move)
            move_number += 1

        node = next_node


def extract_from_pgn_file(pgn_path: Path) -> Iterator[AnnotatedPosition]:
    """Extract annotated positions from a PGN file.

    Args:
        pgn_path: Path to PGN file

    Yields:
        AnnotatedPosition for each meaningful annotation
    """
    try:
        with open(pgn_path, encoding="utf-8", errors="replace") as f:
            game_count = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_count += 1
                try:
                    yield from extract_from_game(game, str(pgn_path))
                except Exception as e:
                    logger.warning(f"Error processing game {game_count} in {pgn_path}: {e}")

            logger.info(f"Processed {game_count} games from {pgn_path.name}")

    except Exception as e:
        logger.error(f"Error reading {pgn_path}: {e}")


def extract_from_directory(
    input_dir: Path,
    output_path: Path,
    recursive: bool = True,
) -> int:
    """Extract annotations from all PGN files in a directory.

    Args:
        input_dir: Directory containing PGN files
        output_path: Path for output JSONL file
        recursive: Search subdirectories

    Returns:
        Number of positions extracted
    """
    pattern = "**/*.pgn" if recursive else "*.pgn"
    pgn_files = list(input_dir.glob(pattern))

    logger.info(f"Found {len(pgn_files)} PGN files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for pgn_path in pgn_files:
            for position in extract_from_pgn_file(pgn_path):
                # Convert to dict and write as JSON line
                data = asdict(position)
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

                if count % 1000 == 0:
                    logger.info(f"Extracted {count} positions...")

    logger.info(f"Total: {count} annotated positions extracted")
    return count


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract annotations from PGN files")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with PGN files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/annotations.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    args = parser.parse_args()

    count = extract_from_directory(
        args.input,
        args.output,
        recursive=not args.no_recursive,
    )

    print(f"Extracted {count} annotated positions to {args.output}")


if __name__ == "__main__":
    main()
