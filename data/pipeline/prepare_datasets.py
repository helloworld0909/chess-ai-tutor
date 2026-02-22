"""Prepare HuggingFace datasets for chess coaching fine-tuning.

Three-phase pipeline:
1. Extract — download & parse HF datasets into intermediate JSONL
2. Augment — add Stockfish analysis (classification, candidates, threats)
3. Format  — compute move facts, format into training messages, split

Usage:
    uv run python data/pipeline/prepare_datasets.py \
        --stockfish /home/zheng/.local/bin/stockfish \
        --output-dir data/processed \
        --max-per-source 0  # 0 = no limit
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chess_mcp.stockfish import Stockfish
from tutor.prompts import SYSTEM_PROMPT, board_ascii, format_user_prompt, move_facts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class RawSample:
    """Intermediate format between extraction and augmentation."""

    fen: str
    move_uci: str
    move_san: str
    coaching_text: str  # the annotation or coaching response
    thinking_text: str  # for <think> block
    source: str


@dataclass
class AugmentedSample:
    """Sample with Stockfish analysis added."""

    fen: str
    move_uci: str
    move_san: str
    coaching_text: str
    thinking_text: str
    source: str
    classification: str
    eval_str: str
    best_move_san: str
    cp_loss: int
    candidates: list[str]  # e.g. ["e4 (+0.35)", "d4 (+0.30)"]
    opponent_threats: list[str]


# ---------------------------------------------------------------------------
# Transformers — one per dataset source
# ---------------------------------------------------------------------------


class BaseTransformer(ABC):
    """Base class for dataset transformers."""

    @abstractmethod
    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        """Yield RawSamples from the dataset source.

        Args:
            max_samples: Maximum samples to extract (0 = no limit).
        """
        ...


class ChessCotTransformer(BaseTransformer):
    """Transform kr4t0n/chess-cot into coaching samples.

    The dataset has: fen, board, reasoning, response (UCI), reward.
    We reframe 'find the best move' reasoning as coaching explanation.
    """

    DATASET_NAME = "kr4t0n/chess-cot"
    MIN_REWARD = 0.5
    MAX_THINKING_LEN = 2048

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        from datasets import load_dataset

        logger.info("Loading %s...", self.DATASET_NAME)
        ds = load_dataset(self.DATASET_NAME, split="train")

        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break

            # Filter by reward quality
            if row.get("reward", 0) < self.MIN_REWARD:
                continue

            fen = row.get("fen", "")
            response = row.get("response", "")
            reasoning = row.get("reasoning", "")

            if not fen or not response:
                continue

            # Validate the move
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(response)
                if move not in board.legal_moves:
                    continue
                san = board.san(move)
            except (ValueError, chess.InvalidMoveError):
                continue

            # Truncate long reasoning for <think> block
            thinking = reasoning[: self.MAX_THINKING_LEN]
            if len(reasoning) > self.MAX_THINKING_LEN:
                thinking += "..."

            # Extract coaching summary from the last 2 sentences of reasoning
            sentences = [s.strip() for s in reasoning.split(".") if s.strip()]
            coaching = ". ".join(sentences[-2:]) + "." if len(sentences) >= 2 else reasoning[-300:]

            yield RawSample(
                fen=fen,
                move_uci=response,
                move_san=san,
                coaching_text=coaching,
                thinking_text=thinking,
                source="chess_cot",
            )
            count += 1

        logger.info("chess-cot: extracted %d samples", count)


class IcannosTransformer(BaseTransformer):
    """Transform Icannos/chess_studies PGN annotations into coaching samples.

    Reuses quality-filtering logic from extract_annotations.py.
    """

    DATASET_NAME = "Icannos/chess_studies"
    MIN_ANNOTATION_LENGTH = 60

    # Instructive keywords (from extract_annotations.py)
    INSTRUCTIVE_KEYWORDS = [
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "in order to",
        "so that",
        "allowing",
        "weak",
        "strong",
        "control",
        "pressure",
        "pawn structure",
        "outpost",
        "file",
        "diagonal",
        "attack",
        "defend",
        "threat",
        "pin",
        "fork",
        "discovery",
        "skewer",
        "sacrifice",
        "exchange",
        "development",
        "center",
        "initiative",
        "tempo",
        "king safety",
        "castle",
        "activity",
        "plan",
        "idea",
        "strategy",
        "positional",
        "advantage",
        "compensation",
        "equality",
        "better",
        "worse",
        "winning",
        "losing",
    ]

    # Lichess board annotation markup (arrows, colored squares) — not instructive text
    _LICHESS_MARKUP_RE = re.compile(r"\[%(?:cal|csl)\s+[^\]]*\]")
    # Patterns that indicate interactive prompts rather than coaching
    _INTERACTIVE_PHRASES = [
        "what would you play",
        "what do you think",
        "can you find",
        "your turn",
        "remember the movements",
        "remember the moves",
        "psst,",
        "with the arrows",
        "as i indicate",
    ]

    def _clean_comment(self, comment: str) -> str:
        """Strip Lichess markup and normalize whitespace."""
        cleaned = self._LICHESS_MARKUP_RE.sub("", comment)
        return " ".join(cleaned.split())

    def _is_quality(self, comment: str) -> bool:
        """Check if annotation is instructive enough."""
        cleaned = self._clean_comment(comment)
        if len(cleaned) < self.MIN_ANNOTATION_LENGTH:
            return False
        lower = cleaned.lower()
        # Reject interactive prompts / questions
        if any(phrase in lower for phrase in self._INTERACTIVE_PHRASES):
            return False
        # Reject annotations that end with a question (quizzes, not coaching)
        if cleaned.rstrip().endswith("?"):
            return False
        # Must contain at least one instructive keyword
        return any(kw in lower for kw in self.INSTRUCTIVE_KEYWORDS)

    def _extract_from_pgn(self, pgn_text: str) -> Iterator[RawSample]:
        """Extract annotated positions from a single PGN string."""
        pgn_io = io.StringIO(pgn_text)

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            board = game.board()
            node = game

            while node.variations:
                next_node = node.variation(0)
                if next_node.move:
                    raw_comment = (next_node.comment or "").strip()
                    comment = self._clean_comment(raw_comment)

                    if self._is_quality(raw_comment):
                        fen = board.fen()
                        move = next_node.move
                        try:
                            san = board.san(move)
                        except Exception:
                            board.push(move)
                            node = next_node
                            continue

                        yield RawSample(
                            fen=fen,
                            move_uci=move.uci(),
                            move_san=san,
                            coaching_text=comment,
                            thinking_text="",
                            source="icannos_studies",
                        )

                    board.push(next_node.move)
                node = next_node

    # Direct CSV URLs (the HF loading script is no longer supported)
    CSV_URLS = [
        "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/lichess_studies.csv",
        "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/others.csv",
    ]

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        import csv

        import httpx

        logger.info("Loading %s (direct CSV download)...", self.DATASET_NAME)

        count = 0
        for url in self.CSV_URLS:
            if max_samples and count >= max_samples:
                break

            logger.info("Downloading %s...", url.split("/")[-1])
            resp = httpx.get(url, follow_redirects=True, timeout=60.0)
            resp.raise_for_status()

            reader = csv.DictReader(io.StringIO(resp.text))
            for row in reader:
                if max_samples and count >= max_samples:
                    break

                pgn_text = row.get("text", "")
                if not pgn_text:
                    continue

                for sample in self._extract_from_pgn(pgn_text):
                    if max_samples and count >= max_samples:
                        break
                    yield sample
                    count += 1

        logger.info("icannos: extracted %d samples", count)


class TextbookTransformer(BaseTransformer):
    """Transform existing augmented.jsonl from the textbook pipeline."""

    def __init__(self, augmented_path: Path):
        self.augmented_path = augmented_path

    def extract(self, max_samples: int = 0) -> Iterator[RawSample]:
        if not self.augmented_path.exists():
            logger.warning("Textbook data not found at %s, skipping", self.augmented_path)
            return

        logger.info("Loading textbook data from %s...", self.augmented_path)
        count = 0

        with open(self.augmented_path, encoding="utf-8") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fen = row.get("fen", "")
                move_uci = row.get("move_uci", "")
                move_san = row.get("move_san", "")
                annotation = row.get("annotation", "")

                if not fen or not move_uci or not annotation:
                    continue

                # Use concepts for thinking text
                concepts = row.get("concepts", [])
                thinking = ""
                if concepts:
                    thinking = f"This position involves {', '.join(concepts)}."

                yield RawSample(
                    fen=fen,
                    move_uci=move_uci,
                    move_san=move_san,
                    coaching_text=annotation,
                    thinking_text=thinking,
                    source="textbook",
                )
                count += 1

        logger.info("textbook: extracted %d samples", count)


# ---------------------------------------------------------------------------
# Stockfish augmentation
# ---------------------------------------------------------------------------


def _classify_cp_loss(cp_loss: int) -> str:
    """Classify move quality by centipawn loss."""
    if cp_loss <= 10:
        return "Best"
    if cp_loss <= 30:
        return "Great"
    if cp_loss <= 80:
        return "Good"
    if cp_loss <= 150:
        return "Inaccuracy"
    if cp_loss <= 300:
        return "Mistake"
    return "Blunder"


def _format_score(value: int, is_mate: bool) -> str:
    """Format a centipawn or mate score for display."""
    if is_mate:
        return f"M{abs(value)}" if value > 0 else f"-M{abs(value)}"
    return f"{value / 100:+.2f}"


async def augment_samples(
    samples: list[RawSample],
    stockfish_path: str,
    depth: int = 16,
    cache_path: Path | None = None,
) -> list[AugmentedSample]:
    """Add Stockfish analysis to raw samples.

    Args:
        samples: Raw extracted samples.
        stockfish_path: Path to Stockfish binary.
        depth: Analysis depth (16 is good balance of speed/accuracy).
        cache_path: Optional JSONL cache for resumable augmentation.

    Returns:
        List of augmented samples.
    """
    # Load cache
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = entry.get("_cache_key", "")
                    if key:
                        cache[key] = entry
                except json.JSONDecodeError:
                    continue
        logger.info("Loaded %d cached augmentations", len(cache))

    engine = Stockfish(path=stockfish_path, depth=depth, threads=1, hash_mb=256)
    await engine.start()

    results: list[AugmentedSample] = []
    cache_file = open(cache_path, "a", encoding="utf-8") if cache_path else None

    try:
        for i, sample in enumerate(samples):
            cache_key = hashlib.md5(f"{sample.fen}:{sample.move_uci}".encode()).hexdigest()

            if cache_key in cache:
                entry = cache[cache_key]
                results.append(
                    AugmentedSample(
                        fen=sample.fen,
                        move_uci=sample.move_uci,
                        move_san=sample.move_san,
                        coaching_text=sample.coaching_text,
                        thinking_text=sample.thinking_text,
                        source=sample.source,
                        classification=entry["classification"],
                        eval_str=entry["eval_str"],
                        best_move_san=entry["best_move_san"],
                        cp_loss=entry["cp_loss"],
                        candidates=entry["candidates"],
                        opponent_threats=entry["opponent_threats"],
                    )
                )
                continue

            try:
                # Run compare_moves for classification
                comparison = await engine.compare_moves(sample.fen, sample.move_uci, depth)
                if "error" in comparison:
                    continue

                cp_loss = comparison.get("cp_loss", 0)
                best_move_uci = comparison.get("best_move", "")

                # Get evaluation score
                analysis = await engine.analyze(sample.fen, depth=depth, multipv=3)
                score = analysis.score
                eval_str = _format_score(
                    score.mate_in if score.mate_in is not None else (score.centipawns or 0),
                    score.mate_in is not None,
                )

                # Convert candidates to SAN
                board = chess.Board(sample.fen)
                candidates = []
                for line in analysis.lines[:3]:
                    try:
                        m = chess.Move.from_uci(line.best_move)
                        s = board.san(m)
                        ls = line.score
                        sc = _format_score(
                            ls.mate_in if ls.mate_in is not None else (ls.centipawns or 0),
                            ls.mate_in is not None,
                        )
                        candidates.append(f"{s} ({sc})")
                    except Exception:
                        pass

                # Convert best move to SAN
                try:
                    best_move_san = board.san(chess.Move.from_uci(best_move_uci))
                except Exception:
                    best_move_san = best_move_uci

                # Get opponent threats
                try:
                    threats_data = await engine.get_threats(sample.fen, depth=depth)
                    threat_moves = threats_data.get("threats", [])
                    opponent_threats = []
                    # Threats are from opponent's perspective — flip turn to parse
                    threat_board = board.copy()
                    threat_board.turn = not threat_board.turn
                    for tm in threat_moves[:3]:
                        try:
                            opponent_threats.append(threat_board.san(chess.Move.from_uci(tm)))
                        except Exception:
                            pass
                except Exception:
                    opponent_threats = []

                classification = _classify_cp_loss(cp_loss)

                aug = AugmentedSample(
                    fen=sample.fen,
                    move_uci=sample.move_uci,
                    move_san=sample.move_san,
                    coaching_text=sample.coaching_text,
                    thinking_text=sample.thinking_text,
                    source=sample.source,
                    classification=classification,
                    eval_str=eval_str,
                    best_move_san=best_move_san,
                    cp_loss=cp_loss,
                    candidates=candidates,
                    opponent_threats=opponent_threats,
                )
                results.append(aug)

                # Write to cache
                if cache_file:
                    cache_entry = {
                        "_cache_key": cache_key,
                        "classification": classification,
                        "eval_str": eval_str,
                        "best_move_san": best_move_san,
                        "cp_loss": cp_loss,
                        "candidates": candidates,
                        "opponent_threats": opponent_threats,
                    }
                    cache_file.write(json.dumps(cache_entry) + "\n")
                    cache_file.flush()

            except Exception as e:
                logger.warning("Error augmenting sample %d: %s", i, e)
                continue

            if (i + 1) % 100 == 0:
                logger.info("Augmented %d / %d samples", i + 1, len(samples))

    finally:
        if cache_file:
            cache_file.close()
        await engine.stop()

    logger.info("Augmentation complete: %d / %d succeeded", len(results), len(samples))
    return results


# ---------------------------------------------------------------------------
# Final formatting
# ---------------------------------------------------------------------------


def format_training_sample(aug: AugmentedSample) -> dict:
    """Convert an augmented sample into the training messages format.

    The user message format is identical to what _llm_comment() sends
    at inference time, via the shared format_user_prompt() function.
    """
    board = chess.Board(aug.fen)
    facts = move_facts(board, chess.Move.from_uci(aug.move_uci))

    user_prompt = format_user_prompt(
        board_ascii_str=board_ascii(board),
        san=aug.move_san,
        classification=aug.classification,
        eval_str=aug.eval_str,
        best_move=aug.best_move_san,
        cp_loss=aug.cp_loss,
        candidates=aug.candidates,
        opponent_threats=aug.opponent_threats,
        facts=facts,
    )

    # Build assistant response with optional thinking
    if aug.thinking_text:
        assistant = f"<think>{aug.thinking_text}</think>\n\n{aug.coaching_text}"
    else:
        assistant = aug.coaching_text

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": aug.source,
            "fen": aug.fen,
            "move_uci": aug.move_uci,
            "classification": aug.classification,
        },
    }


# ---------------------------------------------------------------------------
# Dedup, split, and write
# ---------------------------------------------------------------------------


def dedup_samples(samples: list[AugmentedSample]) -> list[AugmentedSample]:
    """Remove duplicate (fen, move_uci) pairs, keeping the longest coaching text."""
    seen: dict[str, AugmentedSample] = {}
    for s in samples:
        key = f"{s.fen}:{s.move_uci}"
        if key not in seen or len(s.coaching_text) > len(seen[key].coaching_text):
            seen[key] = s
    deduped = list(seen.values())
    logger.info("Dedup: %d → %d samples", len(samples), len(deduped))
    return deduped


def split_and_write(
    samples: list[AugmentedSample],
    output_dir: Path,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """Shuffle, split, format, and write train.jsonl + eval.jsonl.

    Returns:
        Tuple of (train_count, eval_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(samples)

    split_idx = max(1, int(len(samples) * (1 - eval_fraction)))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, subset in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for aug in subset:
                sample = format_training_sample(aug)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(
        "Written %d train + %d eval to %s",
        len(train_samples),
        len(eval_samples),
        output_dir,
    )
    return len(train_samples), len(eval_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def run_pipeline(
    stockfish_path: str,
    output_dir: Path,
    max_per_source: int = 0,
    depth: int = 16,
    textbook_path: Path | None = None,
    skip_augment: bool = False,
) -> None:
    """Run the full extract → augment → format pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract
    logger.info("=== Phase 1: Extract ===")
    all_raw: list[RawSample] = []

    transformers: list[BaseTransformer] = [
        ChessCotTransformer(),
        IcannosTransformer(),
    ]
    if textbook_path:
        transformers.append(TextbookTransformer(textbook_path))

    for tx in transformers:
        try:
            samples = list(tx.extract(max_samples=max_per_source))
            all_raw.extend(samples)
        except Exception as e:
            logger.error("Failed to extract from %s: %s", type(tx).__name__, e)

    logger.info("Total extracted: %d raw samples", len(all_raw))

    if not all_raw:
        logger.error("No samples extracted. Check dataset availability.")
        return

    # Phase 2: Augment with Stockfish
    if skip_augment:
        logger.info("=== Phase 2: Skipped (--skip-augment) ===")
        # Create minimal augmented samples without engine data
        augmented = [
            AugmentedSample(
                fen=s.fen,
                move_uci=s.move_uci,
                move_san=s.move_san,
                coaching_text=s.coaching_text,
                thinking_text=s.thinking_text,
                source=s.source,
                classification="Unknown",
                eval_str="0.00",
                best_move_san=s.move_san,
                cp_loss=0,
                candidates=[],
                opponent_threats=[],
            )
            for s in all_raw
        ]
    else:
        logger.info("=== Phase 2: Augment (%d samples) ===", len(all_raw))
        cache_path = output_dir / ".augment_cache.jsonl"
        augmented = await augment_samples(
            all_raw,
            stockfish_path,
            depth=depth,
            cache_path=cache_path,
        )

    # Phase 3: Format and split
    logger.info("=== Phase 3: Format & Split ===")
    deduped = dedup_samples(augmented)
    train_n, eval_n = split_and_write(deduped, output_dir)

    logger.info("=== Done: %d train + %d eval samples ===", train_n, eval_n)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace datasets for chess coaching fine-tuning"
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: $STOCKFISH_PATH)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for train.jsonl and eval.jsonl",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Max samples per dataset source (0 = unlimited)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Stockfish analysis depth",
    )
    parser.add_argument(
        "--textbook-path",
        type=Path,
        default=None,
        help="Path to augmented.jsonl from the textbook pipeline",
    )
    parser.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip Stockfish augmentation (for testing format only)",
    )

    args = parser.parse_args()

    stockfish_path = args.stockfish or __import__("os").environ.get("STOCKFISH_PATH", "")
    if not stockfish_path and not args.skip_augment:
        parser.error("--stockfish or STOCKFISH_PATH required unless --skip-augment is set")

    asyncio.run(
        run_pipeline(
            stockfish_path=stockfish_path,
            output_dir=args.output_dir,
            max_per_source=args.max_per_source,
            depth=args.depth,
            textbook_path=args.textbook_path,
            skip_augment=args.skip_augment,
        )
    )


if __name__ == "__main__":
    main()
