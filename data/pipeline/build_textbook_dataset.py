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
# Lichess studies downloader (via Icannos HuggingFace CSV dataset)
# ---------------------------------------------------------------------------

# The Lichess search API (/api/study/search) is no longer available.
# We use two complementary sources:
#   1. Icannos/chess_studies HuggingFace CSV dataset (curated subset)
#   2. Direct /api/study/by/{username} fetch for known educational accounts
ICANNOS_CSV_URLS = [
    "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/lichess_studies.csv",
    "https://huggingface.co/datasets/Icannos/chess_studies/resolve/main/others.csv",
]

# Lichess usernames known for publishing high-quality instructional studies.
# Wrong/inactive usernames are safe — the API returns empty content, not an error.
LICHESS_STUDY_AUTHORS = [
    # Streamers / prominent content creators
    "chessbrah",
    "penguingm1",
    "GMHikaruOnTwitch",
    "danya_chess",
    "thechesswebsite",
    "ChessNetwork",
    "imrosen",
    "Saint_Lazarus",
    "GothamChess",
    "BotezLive",
    "AnishOnline",
    "nihalsarin2004",
    "DrDrunkenstein",
    "MagnusCarlsen",
    # Coaches / study authors widely cited on Lichess forums
    "chess-teacher",
    "lovlas",
    "thibault",
    "niklasf",
    "CoachJon",
    "Avetik_Grigoryan",
    "GMBenjamin",
    "GMSmith",
    "IM_Sagar",
    "chess_tempo",
    "yusupov",
    "karteek",
    "kasa_bova",
    "LuckyLooks",
    "VladimirKramnik",
    "MVL",
    "FabianoCaruana",
    "RichardReti",
    "LeninVarela",
    "GM_Illingworth",
    "pepellou",
    "PinkedMink",
    "Mastertan",
    "CalavitoMasters",
    "Chessy_Cat",
    "NimzoRoy",
    "EndgameStudent",
    "LevonAronian",
    "FinnegansWake",
    "ChessOpenings101",
    "Pyrrhox",
]


async def download_lichess_studies(
    workers: int = 8,
) -> list[tuple[str, str]]:
    """Return list of (label, pgn_text) from Icannos Lichess study CSVs."""
    import csv

    results: list[tuple[str, str]] = []

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        timeout=180.0,
        follow_redirects=True,
    ) as client:
        for url in ICANNOS_CSV_URLS:
            csv_name = url.split("/")[-1].replace(".csv", "")
            logger.info("Downloading Icannos CSV: %s ...", url.split("/")[-1])
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", url, e)
                continue

            reader = csv.DictReader(io.StringIO(resp.text))
            row_count = 0
            for row in reader:
                pgn_text = row.get("text", "").strip()
                if pgn_text:
                    results.append((f"lichess_{csv_name}_{row_count}", pgn_text))
                    row_count += 1
            logger.info("  %s: %d PGN entries loaded", csv_name, row_count)

    logger.info("Loaded %d Lichess study PGN entries from Icannos CSVs", len(results))
    return results


async def _fetch_lichess_author_studies(
    client: httpx.AsyncClient,
    username: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Fetch all public studies for a Lichess user as bulk PGN."""
    async with semaphore:
        try:
            resp = await client.get(
                f"https://lichess.org/api/study/by/{username}.pgn",
                timeout=120.0,
            )
            if resp.status_code == 200 and resp.text.strip():
                return username, resp.text
            return username, None
        except Exception as e:
            logger.warning("lichess author %s: %s", username, e)
            return username, None


async def download_lichess_author_studies(workers: int = 4) -> list[tuple[str, str]]:
    """Fetch all public studies from LICHESS_STUDY_AUTHORS via /api/study/by/{user}."""
    semaphore = asyncio.Semaphore(workers)
    results: list[tuple[str, str]] = []

    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        timeout=180.0,
        follow_redirects=True,
    ) as client:
        tasks = [
            _fetch_lichess_author_studies(client, user, semaphore) for user in LICHESS_STUDY_AUTHORS
        ]
        raw = await asyncio.gather(*tasks)

    for username, pgn in raw:
        if pgn:
            results.append((f"lichess_author_{username}", pgn))
            logger.info("  %s: fetched", username)

    logger.info(
        "Fetched studies from %d / %d Lichess authors",
        len(results),
        len(LICHESS_STUDY_AUTHORS),
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def build(
    output_path: Path,
    pgnmentor_workers: int = 12,
    lichess_workers: int = 8,
    max_positions_per_file: int = 500,
    skip_pgnmentor: bool = False,
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

    logger.info("After Lichess CSV: %d total positions", total_written)

    # ---- Phase 1b: Lichess author studies ----
    logger.info(
        "=== Phase 1b: Lichess Author Studies (%d accounts) ===", len(LICHESS_STUDY_AUTHORS)
    )
    author_collections = await download_lichess_author_studies(workers=max(1, lichess_workers // 2))
    for label, pgn_text in author_collections:
        positions = list(_parse_pgn_annotations(pgn_text, source=label))
        n = write_positions(positions)
        if n:
            logger.info("  %s → %d positions", label, n)
            total_written += n

    logger.info("After author studies: %d total positions", total_written)

    # ---- Phase 2: PGN Mentor (rarely annotated — skip by default) ----
    if skip_pgnmentor:
        logger.info("=== Phase 2: PGN Mentor skipped ===")
        out.close()
        logger.info("=== Done: %d total positions in %s ===", total_written, output_path)
        return
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
    parser.add_argument(
        "--skip-pgnmentor",
        action="store_true",
        default=True,
        help="Skip PGN Mentor phase (default: True — PGN Mentor games are rarely annotated)",
    )
    parser.add_argument(
        "--pgnmentor",
        dest="skip_pgnmentor",
        action="store_false",
        help="Enable PGN Mentor phase",
    )
    args = parser.parse_args()

    asyncio.run(
        build(
            output_path=args.output,
            pgnmentor_workers=args.workers,
            lichess_workers=min(args.workers, 8),
            max_positions_per_file=args.max_per_file,
            skip_pgnmentor=args.skip_pgnmentor,
        )
    )


if __name__ == "__main__":
    main()
