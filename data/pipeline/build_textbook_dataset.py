"""Build annotated chess textbook dataset from PGN sources.

Pipeline:
  1. Download PGN files from PGN Mentor (players, events, openings)
  2. Fetch annotated Lichess studies via NDJSON bulk API
  3. Parse PGN move comments → (FEN, move, annotation) triples
  4. Quality-filter annotations and extract chess concepts
  5. Write data/raw/textbook_augmented.jsonl

The JSONL schema is compatible with TextbookTransformer in prepare_datasets.py:
  {"fen": "...", "move_uci": "...", "move_san": "...",
   "annotation": "...", "concepts": [...], "source": "..."}

Usage:
    uv run python data/pipeline/build_textbook_dataset.py \
        --output data/raw/textbook_augmented.jsonl \
        --workers 12
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PGNMENTOR_BASE = "https://pgnmentor.com"

# Minimum annotation length to keep (characters)
MIN_ANNOTATION_LEN = 30

# Chess concepts for the thinking field — matches keywords in annotation text
CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "tactics": [
        "pin",
        "fork",
        "skewer",
        "discovery",
        "discovered",
        "double check",
        "sacrifice",
        "combination",
        "zwischenzug",
        "in-between",
        "deflection",
        "decoy",
        "removing the defender",
        "back rank",
    ],
    "strategy": [
        "pawn structure",
        "outpost",
        "weak pawn",
        "isolated pawn",
        "doubled pawn",
        "passed pawn",
        "majority",
        "minority attack",
        "good bishop",
        "bad bishop",
        "knight vs bishop",
        "two bishops",
        "open file",
        "half-open",
        "seventh rank",
        "activity",
    ],
    "endgame": [
        "king and pawn",
        "opposition",
        "zugzwang",
        "triangulation",
        "fortress",
        "lucena",
        "philidor",
        "rook endgame",
        "pawn ending",
        "knight endgame",
        "bishop endgame",
        "queen endgame",
    ],
    "opening": [
        "development",
        "center control",
        "castling",
        "gambit",
        "opening theory",
        "transposition",
        "main line",
        "sideline",
        "initiative",
        "tempo",
        "compensation",
    ],
    "positional": [
        "space",
        "weak square",
        "strong square",
        "prophylaxis",
        "restriction",
        "overprotection",
        "coordination",
        "imbalance",
    ],
    "attack": [
        "king attack",
        "kingside attack",
        "pawn storm",
        "piece sacrifice",
        "mating attack",
        "checkmate",
        "forced",
        "winning",
    ],
    "defense": [
        "counterplay",
        "active defense",
        "passive defense",
        "exchange",
        "simplification",
        "drawing",
        "perpetual",
    ],
}

INSTRUCTIVE_KEYWORDS = [
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "in order to",
    "so that",
    "allowing",
    "preventing",
    "threatening",
    "winning",
    "losing",
    "better",
    "worse",
    "weak",
    "strong",
    "control",
    "attack",
    "defend",
    "plan",
    "idea",
    "strategy",
]

_LICHESS_MARKUP_RE = re.compile(r"\[%(?:cal|csl)\s+[^\]]*\]|\[%[^\]]*\]")
_EVAL_TAG_RE = re.compile(r"\[%eval[^\]]*\]")
_CLOCK_TAG_RE = re.compile(r"\[%clk[^\]]*\]")
_MOVE_NUM_RE = re.compile(r"^\d+\.+\s*")


# ---------------------------------------------------------------------------
# Annotation extraction from PGN
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedPosition:
    fen: str
    move_uci: str
    move_san: str
    annotation: str
    concepts: list[str]
    source: str


def _clean_comment(comment: str) -> str:
    """Strip PGN markup tags and normalize whitespace."""
    comment = _LICHESS_MARKUP_RE.sub("", comment)
    comment = _EVAL_TAG_RE.sub("", comment)
    comment = _CLOCK_TAG_RE.sub("", comment)
    comment = _MOVE_NUM_RE.sub("", comment)
    return " ".join(comment.split()).strip()


def _is_quality(text: str) -> bool:
    """Return True if annotation is instructive and long enough."""
    if len(text) < MIN_ANNOTATION_LEN:
        return False
    lower = text.lower()
    if lower.rstrip().endswith("?"):
        return False  # quiz, not coaching
    interactive = [
        "what would you play",
        "what do you think",
        "can you find",
        "your turn",
        "find the best",
        "solve this",
    ]
    if any(phrase in lower for phrase in interactive):
        return False
    return any(kw in lower for kw in INSTRUCTIVE_KEYWORDS)


def _extract_concepts(text: str) -> list[str]:
    lower = text.lower()
    return [
        concept
        for concept, keywords in CONCEPT_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]


def _parse_pgn_annotations(
    pgn_text: str,
    source: str,
    max_per_game: int = 8,
) -> Iterator[AnnotatedPosition]:
    """Yield annotated positions from PGN text."""
    pgn_io = io.StringIO(pgn_text)
    games_parsed = 0

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            break
        if game is None:
            break

        games_parsed += 1
        board = game.board()
        count = 0

        # Build a source label from game headers — skip if headers are unknown
        event = game.headers.get("Event", "")
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        year = game.headers.get("Date", "")[:4]
        # Skip games with missing/placeholder headers
        if "?" in (white + black + event) or not (white or black or event):
            continue
        game_label = f"{white} vs {black}, {event} {year}".strip(", ")

        node = game
        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            if move is None:
                node = next_node
                continue

            comment = next_node.comment or ""
            san = board.san(move)
            fen_before = board.fen()

            annotation = _clean_comment(comment)

            if annotation and _is_quality(annotation):
                concepts = _extract_concepts(annotation)
                yield AnnotatedPosition(
                    fen=fen_before,
                    move_uci=move.uci(),
                    move_san=san,
                    annotation=annotation,
                    concepts=concepts,
                    source=source,
                )
                count += 1
                if count >= max_per_game:
                    break
            elif not annotation and game_label:
                # Unannotated: emit with context placeholder so LLM coach rewrites it
                # Only for games with recognizable context (events, known players)
                if event and (white or black):
                    annotation = f"This move was played in {game_label}."
                    concepts = []
                    yield AnnotatedPosition(
                        fen=fen_before,
                        move_uci=move.uci(),
                        move_san=san,
                        annotation=annotation,
                        concepts=concepts,
                        source=source,
                    )
                    count += 1
                    if count >= max_per_game:
                        break

            board.push(move)
            node = next_node


# ---------------------------------------------------------------------------
# PGN Mentor downloader
# ---------------------------------------------------------------------------


async def _fetch_pgnmentor_file(
    client: httpx.AsyncClient,
    relative_url: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Download a single PGN Mentor file, return PGN text."""
    url = f"{PGNMENTOR_BASE}/{relative_url}"
    async with semaphore:
        try:
            resp = await client.get(url, follow_redirects=True, timeout=60.0)
            if resp.status_code != 200:
                logger.warning("pgnmentor %s → %d", relative_url, resp.status_code)
                return None
        except Exception as e:
            logger.warning("pgnmentor %s error: %s", relative_url, e)
            return None

    if relative_url.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                parts: list[str] = []
                for name in zf.namelist():
                    if name.lower().endswith(".pgn"):
                        parts.append(zf.read(name).decode("utf-8", errors="replace"))
                return "\n\n".join(parts) if parts else None
        except Exception as e:
            logger.warning("pgnmentor zip error %s: %s", relative_url, e)
            return None
    else:
        return resp.text


async def download_pgnmentor(
    file_list: list[str],
    workers: int = 10,
) -> Iterator[tuple[str, str]]:
    """Yield (source_label, pgn_text) for each PGN Mentor file."""
    semaphore = asyncio.Semaphore(workers)
    results: list[tuple[str, str | None]] = []

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        timeout=90.0,
    ) as client:
        tasks = [_fetch_pgnmentor_file(client, url, semaphore) for url in file_list]
        raw = await asyncio.gather(*tasks)
        results = list(zip(file_list, raw))

    for url, text in results:
        if text:
            label = Path(url).stem  # e.g. "Kasparov", "Chennai2024"
            section = url.split("/")[0]  # players/events/openings
            yield f"pgnmentor_{section}_{label}", text


# ---------------------------------------------------------------------------
# Lichess studies downloader
# ---------------------------------------------------------------------------

# Instructive study topics to query
LICHESS_STUDY_QUERIES = [
    # Endgames
    "rook endgame technique",
    "king and pawn endgame",
    "bishop endgame",
    "knight endgame",
    "queen endgame",
    "pawn endgame theory",
    "rook vs pawn",
    "lucena philidor",
    "zugzwang opposition",
    # Tactics
    "tactical patterns fork pin skewer",
    "discovered attack back rank mate",
    "sacrifice combination chess",
    "deflection decoy chess tactics",
    # Strategy
    "pawn structure strategy",
    "isolated queen pawn",
    "passed pawn",
    "good bishop bad bishop",
    "open file rook",
    "outpost knight",
    "minority attack majority",
    "prophylaxis restriction",
    # Openings
    "opening principles development",
    "sicilian defense theory",
    "french defense strategy",
    "caro kann theory",
    "king indian defense",
    "queen gambit declined",
    "ruy lopez theory",
    "english opening",
    "nimzo indian theory",
    "queens gambit",
    # Other
    "checkmate patterns",
    "chess fundamentals beginners",
    "middlegame planning",
    "chess strategy masterclass",
    "endgame studies puzzles",
]

# Well-known Lichess users who publish instructional studies
LICHESS_STUDY_AUTHORS = [
    "chessbrah",
    "penguingm1",
    "GMHikaruOnTwitch",
    "danya_chess",
    "Saint_Lazarus",
    "thechesswebsite",
    "ChessNetwork",
]

# Hand-picked high-quality study IDs (Lichess public studies)
LICHESS_STUDY_IDS_SEED = [
    "4Km7XB3Y",  # Endgame fundamentals
    "7CL38Nfz",  # Rook endgames
    "GlpFoAMX",  # Pawn endgames
    "Yl7FDSEl",  # Common pawn structures
    "PqMfUgLT",  # Chess tactics course
]


async def _search_lichess_studies(
    client: httpx.AsyncClient,
    query: str,
    nb: int = 20,
) -> list[str]:
    """Return study IDs matching query."""
    try:
        resp = await client.get(
            "https://lichess.org/api/study/search",
            params={"q": query, "nb": nb},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return []
        ids = []
        for line in resp.text.splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        ids.append(obj["id"])
                except json.JSONDecodeError:
                    pass
        return ids
    except Exception:
        return []


async def _fetch_lichess_study(
    client: httpx.AsyncClient,
    study_id: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Fetch a Lichess study as PGN."""
    async with semaphore:
        try:
            resp = await client.get(
                f"https://lichess.org/api/study/{study_id}.pgn",
                timeout=30.0,
            )
            if resp.status_code == 200:
                return resp.text
            return None
        except Exception:
            return None


async def _fetch_lichess_author_studies(
    client: httpx.AsyncClient,
    username: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Fetch all studies by a Lichess user as bulk PGN."""
    async with semaphore:
        try:
            resp = await client.get(
                f"https://lichess.org/api/study/by/{username}.pgn",
                timeout=60.0,
            )
            if resp.status_code == 200:
                return resp.text
            return None
        except Exception:
            return None


async def download_lichess_studies(
    workers: int = 8,
) -> list[tuple[str, str]]:
    """Return list of (label, pgn_text) from Lichess studies."""
    semaphore = asyncio.Semaphore(workers)
    results: list[tuple[str, str]] = []

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        timeout=60.0,
    ) as client:
        # 1. Collect study IDs via search
        logger.info("Searching Lichess studies (%d queries)...", len(LICHESS_STUDY_QUERIES))
        search_tasks = [_search_lichess_studies(client, q, nb=30) for q in LICHESS_STUDY_QUERIES]
        search_results = await asyncio.gather(*search_tasks)
        all_ids: set[str] = set(LICHESS_STUDY_IDS_SEED)
        for ids in search_results:
            all_ids.update(ids)
        logger.info("Found %d unique study IDs", len(all_ids))

        # 2. Fetch each study
        fetch_tasks = [_fetch_lichess_study(client, sid, semaphore) for sid in all_ids]
        pgns = await asyncio.gather(*fetch_tasks)
        for sid, pgn in zip(all_ids, pgns):
            if pgn:
                results.append((f"lichess_study_{sid}", pgn))
            await asyncio.sleep(0.05)  # gentle rate limit

        # 3. Fetch author studies
        logger.info("Fetching studies from %d authors...", len(LICHESS_STUDY_AUTHORS))
        author_tasks = [
            _fetch_lichess_author_studies(client, user, semaphore) for user in LICHESS_STUDY_AUTHORS
        ]
        author_pgns = await asyncio.gather(*author_tasks)
        for user, pgn in zip(LICHESS_STUDY_AUTHORS, author_pgns):
            if pgn:
                results.append((f"lichess_author_{user}", pgn))

    logger.info("Downloaded %d Lichess study PGN collections", len(results))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def build(
    output_path: Path,
    pgnmentor_workers: int = 12,
    lichess_workers: int = 8,
    max_positions_per_file: int = 500,
) -> None:
    """Run full textbook data collection pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing entries to skip duplicates (resumable)
    seen_keys: set[str] = set()
    existing = 0
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    seen_keys.add(f"{row['fen']}:{row['move_uci']}")
                    existing += 1
                except Exception:
                    pass
        logger.info("Resuming: %d existing entries", existing)

    total_written = existing
    out = open(output_path, "a", encoding="utf-8")

    def write_positions(positions: list[AnnotatedPosition]) -> int:
        written = 0
        for pos in positions:
            key = f"{pos.fen}:{pos.move_uci}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.write(
                json.dumps(
                    {
                        "fen": pos.fen,
                        "move_uci": pos.move_uci,
                        "move_san": pos.move_san,
                        "annotation": pos.annotation,
                        "concepts": pos.concepts,
                        "source": pos.source,
                    }
                )
                + "\n"
            )
            written += 1
        if written:
            out.flush()
        return written

    # ---- Phase 1: Lichess studies (annotated, high quality) ----
    logger.info("=== Phase 1: Lichess Studies ===")
    lichess_collections = await download_lichess_studies(workers=lichess_workers)
    for label, pgn_text in lichess_collections:
        positions = list(_parse_pgn_annotations(pgn_text, source=label))
        n = write_positions(positions)
        if n:
            logger.info("  %s → %d positions", label, n)
            total_written += n

    logger.info("After Lichess: %d total positions", total_written)

    # ---- Phase 2: PGN Mentor ----
    logger.info("=== Phase 2: PGN Mentor (%d files) ===", 0)

    # Load the full file list
    pgnmentor_files_path = Path(__file__).parent / "pgnmentor_files.txt"
    if pgnmentor_files_path.exists():
        all_files = [l.strip() for l in pgnmentor_files_path.read_text().splitlines() if l.strip()]
    else:
        # Fetch dynamically
        logger.info("Fetching PGN Mentor file list...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{PGNMENTOR_BASE}/files.html")
        all_files_raw = re.findall(
            r'href="((?:players|events|openings)/[^"]+\.(?:pgn|zip))"', resp.text
        )
        all_files = sorted(set(all_files_raw))
        pgnmentor_files_path.write_text("\n".join(all_files))

    logger.info("PGN Mentor: %d files to process", len(all_files))

    # Process in batches to limit memory usage
    BATCH = 50
    for batch_start in range(0, len(all_files), BATCH):
        batch = all_files[batch_start : batch_start + BATCH]
        logger.info("  Batch %d-%d / %d", batch_start, batch_start + len(batch), len(all_files))

        semaphore = asyncio.Semaphore(pgnmentor_workers)
        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            timeout=90.0,
        ) as client:
            raw = await asyncio.gather(
                *[_fetch_pgnmentor_file(client, url, semaphore) for url in batch]
            )

        batch_written = 0
        for url, pgn_text in zip(batch, raw):
            if not pgn_text:
                continue
            label = f"pgnmentor_{Path(url).stem}"
            positions = list(_parse_pgn_annotations(pgn_text, source=label, max_per_game=5))[
                :max_positions_per_file
            ]
            n = write_positions(positions)
            batch_written += n

        total_written += batch_written
        logger.info("  Batch wrote %d positions (total: %d)", batch_written, total_written)
        await asyncio.sleep(1.0)  # pause between batches

    out.close()
    logger.info("=== Done: %d total positions in %s ===", total_written, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build textbook chess dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/raw/textbook_augmented.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=12, help="Concurrent HTTP workers")
    parser.add_argument(
        "--max-per-file", type=int, default=500, help="Max positions extracted per PGN file"
    )
    args = parser.parse_args()

    asyncio.run(
        build(
            output_path=args.output,
            pgnmentor_workers=args.workers,
            lichess_workers=min(args.workers, 8),
            max_positions_per_file=args.max_per_file,
        )
    )


if __name__ == "__main__":
    main()
