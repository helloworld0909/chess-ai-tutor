"""Download and organize annotated chess PGN collections.

Sources:
- PGN Mentor: https://pgnmentor.com/files.html
- Lichess Studies API: https://lichess.org/api
- Lichess Elite DB: https://database.nikonoel.fr/
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PGNSource:
    """A source for PGN files."""

    name: str
    url: str
    annotated: bool = True
    format: str = "pgn"


# Known sources for annotated PGNs
PGNMENTOR_PLAYERS = [
    "Kasparov",
    "Carlsen",
    "Fischer",
    "Capablanca",
    "Alekhine",
    "Tal",
    "Petrosian",
    "Botvinnik",
    "Karpov",
    "Anand",
    "Kramnik",
    "Topalov",
    "Aronian",
    "Caruana",
    "Nakamura",
    "Giri",
]

PGNMENTOR_SOURCES = [
    PGNSource(
        name=f"pgnmentor_{player.lower()}",
        url=f"https://pgnmentor.com/players/{player}.zip",
        annotated=True,
    )
    for player in PGNMENTOR_PLAYERS
]

# Classic collections
CLASSIC_SOURCES = [
    PGNSource(
        name="morphy_games",
        url="https://pgnmentor.com/players/Morphy.zip",
        annotated=True,
    ),
]


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    chunk_size: int = 8192,
) -> bool:
    """Download a file from URL.

    Args:
        client: HTTP client
        url: URL to download
        output_path: Where to save
        chunk_size: Download chunk size

    Returns:
        True if successful
    """
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != 200:
                logger.warning(f"Failed to download {url}: {response.status_code}")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)

            logger.info(f"Downloaded: {output_path.name}")
            return True

    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


async def download_pgnmentor_games(
    output_dir: Path,
    players: list[str] | None = None,
) -> list[Path]:
    """Download games from PGN Mentor.

    Args:
        output_dir: Directory to save files
        players: List of player names (defaults to PGNMENTOR_PLAYERS)

    Returns:
        List of downloaded file paths
    """
    players = players or PGNMENTOR_PLAYERS
    downloaded = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for player in players:
            url = f"https://pgnmentor.com/players/{player}.zip"
            output_path = output_dir / f"{player.lower()}.zip"

            if output_path.exists():
                logger.info(f"Skipping {player} (already exists)")
                downloaded.append(output_path)
                continue

            if await download_file(client, url, output_path):
                downloaded.append(output_path)

            # Be nice to the server
            await asyncio.sleep(1)

    return downloaded


async def fetch_lichess_study(
    client: httpx.AsyncClient,
    study_id: str,
) -> str | None:
    """Fetch a Lichess study as PGN.

    Args:
        client: HTTP client
        study_id: Lichess study ID

    Returns:
        PGN content or None
    """
    url = f"https://lichess.org/api/study/{study_id}.pgn"

    try:
        response = await client.get(url)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Failed to fetch study {study_id}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching study {study_id}: {e}")
        return None


async def search_lichess_studies(
    query: str,
    limit: int = 100,
) -> list[dict]:
    """Search for Lichess studies.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of study metadata
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = "https://lichess.org/api/study/search"
        params = {"q": query, "nb": limit}

        try:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                # Response is newline-delimited JSON
                studies = []
                for line in response.text.strip().split("\n"):
                    if line:
                        import json

                        studies.append(json.loads(line))
                return studies
            return []
        except Exception as e:
            logger.error(f"Error searching studies: {e}")
            return []


async def download_lichess_studies(
    output_dir: Path,
    study_ids: list[str] | None = None,
    search_queries: list[str] | None = None,
) -> list[Path]:
    """Download Lichess studies as PGN.

    Args:
        output_dir: Directory to save files
        study_ids: Specific study IDs to download
        search_queries: Queries to search for studies

    Returns:
        List of downloaded file paths
    """
    downloaded = []
    all_study_ids = list(study_ids or [])

    # Search for additional studies
    if search_queries:
        for query in search_queries:
            studies = await search_lichess_studies(query)
            for study in studies:
                if "id" in study:
                    all_study_ids.append(study["id"])

    # Remove duplicates
    all_study_ids = list(set(all_study_ids))
    logger.info(f"Downloading {len(all_study_ids)} Lichess studies")

    async with httpx.AsyncClient(timeout=30.0) as client:
        for study_id in all_study_ids:
            output_path = output_dir / f"lichess_study_{study_id}.pgn"

            if output_path.exists():
                downloaded.append(output_path)
                continue

            pgn = await fetch_lichess_study(client, study_id)
            if pgn:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(pgn)
                downloaded.append(output_path)
                logger.info(f"Downloaded study: {study_id}")

            # Rate limiting
            await asyncio.sleep(0.5)

    return downloaded


def extract_zip_files(directory: Path) -> list[Path]:
    """Extract all ZIP files in a directory.

    Args:
        directory: Directory containing ZIP files

    Returns:
        List of extracted PGN file paths
    """
    import zipfile

    extracted = []

    for zip_path in directory.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".pgn"):
                        output_path = directory / name
                        if not output_path.exists():
                            zf.extract(name, directory)
                            logger.info(f"Extracted: {name}")
                        extracted.append(output_path)
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")

    return extracted


async def collect_all(
    output_dir: Path,
    include_pgnmentor: bool = True,
    include_lichess: bool = True,
    lichess_queries: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Collect PGNs from all sources.

    Args:
        output_dir: Base output directory
        include_pgnmentor: Include PGN Mentor downloads
        include_lichess: Include Lichess studies
        lichess_queries: Search queries for Lichess

    Returns:
        Dictionary of source -> file paths
    """
    result = {}

    if include_pgnmentor:
        pgnmentor_dir = output_dir / "pgnmentor"
        await download_pgnmentor_games(pgnmentor_dir)
        pgn_files = extract_zip_files(pgnmentor_dir)
        result["pgnmentor"] = pgn_files

    if include_lichess:
        lichess_dir = output_dir / "lichess"
        default_queries = lichess_queries or [
            "openings fundamentals",
            "endgame technique",
            "tactics training",
            "pawn structure",
            "positional chess",
        ]
        lichess_files = await download_lichess_studies(
            lichess_dir,
            search_queries=default_queries,
        )
        result["lichess"] = lichess_files

    return result


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect chess PGN files")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-pgnmentor",
        action="store_true",
        help="Skip PGN Mentor downloads",
    )
    parser.add_argument(
        "--no-lichess",
        action="store_true",
        help="Skip Lichess downloads",
    )
    parser.add_argument(
        "--lichess-query",
        "-q",
        action="append",
        help="Lichess search queries",
    )

    args = parser.parse_args()

    result = asyncio.run(
        collect_all(
            args.output,
            include_pgnmentor=not args.no_pgnmentor,
            include_lichess=not args.no_lichess,
            lichess_queries=args.lichess_query,
        )
    )

    total = sum(len(files) for files in result.values())
    logger.info(f"Collected {total} PGN files")

    for source, files in result.items():
        logger.info(f"  {source}: {len(files)} files")


if __name__ == "__main__":
    main()
