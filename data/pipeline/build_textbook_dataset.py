"""Build annotated chess textbook dataset from Lichess studies.

Pipeline:
  1. Fetch Icannos/chess_studies CSV from HuggingFace (pre-curated Lichess studies)
  2. BFS crawl Lichess /api/study/by/{user}/export.pgn for known accounts + discovered authors
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator

import chess
import chess.pgn
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)  # suppress illegal-move parse noise

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


_RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|\*)\s*$")
_PURE_MOVE_RE = re.compile(r"^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?$")


def _is_quality(text: str) -> bool:
    """Return True if annotation has enough content to be worth coaching.

    Deliberately permissive — the LLM will filter weak cases at coaching time
    (instructed to return SKIP for useless annotations).  We only drop truly
    empty / pure-result / single-word entries here.
    """
    if len(text) < 15:
        return False
    if _RESULT_RE.match(text.strip()):
        return False
    # Pure move notation with no prose ("Nxe5+", "O-O-O") — no coaching value
    if _PURE_MOVE_RE.match(text.strip()):
        return False
    return True


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
) -> Iterator[AnnotatedPosition]:
    """Yield annotated positions from PGN text."""
    pgn_io = io.StringIO(pgn_text)

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            break
        if game is None:
            break

        board = game.board()
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

            board.push(move)
            node = next_node


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
    "MatoJelic",
    "agadmator",
    "HangingPawns",
    "GingerGM",
    "JohnBartholomew",
    "ChessTalk",
    "PowerPlayChess",
    "ChessDiagrams",
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
    # Additional coaches and educators
    "IM_Silman",
    "GM_Smirnov",
    "chessmood",
    "ChessKidOfficial",
    "dp_chess",
    "chess_art",
    "NMDanielRensch",
    "Finegold",
    "Ben_Finegold",
    "gserper",
    "Shellder",
    "Chess_Fundamentals",
    "liChessCoach",
    "MrChessTeacher",
    "Kostya_Kavutskiy",
    "kkavut",
    "NicoZwirs",
    "colovic",
    "Chess_Explained",
    "IgorSmirnov",
    "ChessHeroes",
    "FM_Michael_Rahal",
    "WGM_Natalia_Pogonina",
    "ShahadeChess",
    "JenShahade",
    "USChessLeague",
]

# Lichess teams whose members are likely to include quality annotators.
# 404s are silently skipped.
LICHESS_STUDY_TEAMS = [
    "lichess-streamer",
    "lichess-coaches",
    "chess-teachers",
    "chess-coach",
    "chess-training",
    "lichess-study-group",
    "chess-university",
    "the-chess-masters",
    "chess-improvement",
    "chess-openings",
    "endgame-practice",
    "chess-tactics",
]

# Study topic slugs to mine for author usernames.
LICHESS_STUDY_TOPICS = [
    "Endgames",
    "Opening",
    "Tactics",
    "Middlegame",
    "Strategy",
    "Checkmate",
    "Pawn Structure",
    "Attacking Chess",
    "Chess Fundamentals",
    "Positional Chess",
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


def _extract_annotators_from_pgn(pgn_text: str) -> set[str]:
    """Extract Lichess usernames from [Annotator] tags in PGN text.

    Lichess exports annotators as full profile URLs:
        [Annotator "https://lichess.org/@/username"]
    but sometimes also as bare usernames.
    """
    found: set[str] = set()
    for m in re.finditer(r'\[Annotator "([^"]+)"\]', pgn_text):
        value = m.group(1).strip()
        # URL form: https://lichess.org/@/username
        url_m = re.search(r"lichess\.org/@/([A-Za-z0-9_-]{2,30})", value)
        if url_m:
            found.add(url_m.group(1))
        elif " " not in value and "," not in value and "@" not in value and len(value) <= 30:
            found.add(value)
    return found


async def _fetch_one_user(
    client: httpx.AsyncClient,
    username: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Fetch all public studies for a Lichess user via /api/study/by/{user}/export.pgn."""
    import random

    url = f"https://lichess.org/api/study/by/{username}/export.pgn"
    for attempt in range(5):
        async with semaphore:
            try:
                resp = await client.get(url, timeout=120.0)
                if resp.status_code == 200 and resp.text.strip().startswith("["):
                    return username, resp.text
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 0))
                    # Full jitter: uniform sample in [0, cap] so concurrent workers
                    # spread out rather than thundering-herd at the same moment.
                    cap = min(2**attempt * 2, 120)
                    wait = max(retry_after, random.uniform(0, cap))
                    logger.warning(
                        "lichess/%s: 429 — backoff %.1fs (attempt %d)", username, wait, attempt + 1
                    )
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code not in (404, 400):
                    logger.debug("lichess/%s: status %d", username, resp.status_code)
                return username, None
            except Exception as e:
                cap = min(2**attempt * 2, 30)
                if attempt < 4:
                    await asyncio.sleep(random.uniform(0, cap))
                else:
                    logger.warning("lichess/%s: %s", username, e)
    return username, None


_CHESS_TITLES = frozenset({"GM", "IM", "FM", "CM", "NM", "WGM", "WIM", "WFM", "WCM", "LM"})
_MIN_RATING = 2000
_MIN_FOLLOWERS = 100


async def _filter_users_by_quality(
    client: httpx.AsyncClient,
    usernames: list[str],
) -> list[str]:
    """Return only usernames that pass the quality bar via Lichess bulk user API.

    Criteria (OR):
      - Has a chess title (GM/IM/FM/CM/NM/W*/LM)
      - Best classical/rapid/blitz rating >= 2000
      - followersCount >= 100
    Falls back to accepting all users if the API call fails.
    """
    if not usernames:
        return []
    try:
        resp = await client.post(
            "https://lichess.org/api/users",
            content=",".join(usernames),
            headers={"Content-Type": "text/plain"},
            timeout=30.0,
        )
        if resp.status_code != 200:
            return usernames
        users = resp.json()
    except Exception:
        return usernames  # permissive fallback

    qualified: list[str] = []
    for u in users:
        uid: str = u.get("username", "")
        title: str = u.get("title") or ""
        followers: int = u.get("followersCount", 0)
        perfs = u.get("perfs", {})
        best_rating = max(
            (perfs.get(k, {}).get("rating", 0) for k in ("classical", "rapid", "blitz")),
            default=0,
        )
        ok = title in _CHESS_TITLES or best_rating >= _MIN_RATING or followers >= _MIN_FOLLOWERS
        if ok:
            logger.info(
                "  quality OK: %s (title=%s, rating=%d, followers=%d)",
                uid,
                title,
                best_rating,
                followers,
            )
            qualified.append(uid)
        else:
            logger.debug(
                "  quality skip: %s (title=%s, rating=%d, followers=%d)",
                uid,
                title,
                best_rating,
                followers,
            )
    return qualified


async def crawl_lichess_studies(
    token: str,
    seed_users: list[str],
    workers: int = 4,
    max_users: int = 2000,
    crawled_users_path: Path | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """BFS crawl Lichess studies starting from seed_users.

    After fetching each user's studies, extracts [Annotator] usernames from
    the returned PGN and enqueues them for fetching too.  Stops when the queue
    is empty or max_users is reached.

    crawled_users_path: if given, already-fetched usernames are loaded from this
    file on startup (skipping re-fetch) and each newly-fetched user is appended
    to it, so restarts skip completed work.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "chess-ai-tutor/1.0 (educational; github.com/chess-ai)",
    }

    # Load persisted completed users to skip on restart
    already_done: set[str] = set()
    if crawled_users_path and crawled_users_path.exists():
        with open(crawled_users_path, encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    already_done.add(u.lower())
        logger.info(
            "Loaded %d already-crawled users from %s", len(already_done), crawled_users_path
        )

    crawled_users_file = (
        open(crawled_users_path, "a", encoding="utf-8") if crawled_users_path else None
    )

    visited: set[str] = set(already_done)
    queue: list[str] = []
    for u in seed_users:
        key = u.lower()
        if key not in visited:
            visited.add(key)
            queue.append(u)

    with_studies = 0
    semaphore = asyncio.Semaphore(workers)

    try:
        async with httpx.AsyncClient(
            headers=headers, timeout=180.0, follow_redirects=True
        ) as client:
            while queue and len(visited) <= max_users:
                # Process one batch at a time so we can yield + write immediately
                batch, queue = queue[:workers], queue[workers:]
                raw = await asyncio.gather(*[_fetch_one_user(client, u, semaphore) for u in batch])
                await asyncio.sleep(0.1)  # small pause between batches
                for username, pgn in raw:
                    # Persist completion regardless of whether the user had studies
                    if crawled_users_file:
                        crawled_users_file.write(username + "\n")
                        crawled_users_file.flush()
                    if not pgn:
                        continue
                    with_studies += 1
                    logger.info(
                        "  [%d visited / %d queued] %s: %d chars",
                        len(visited),
                        len(queue),
                        username,
                        len(pgn),
                    )
                    yield f"lichess_crawl_{username}", pgn
                    # Discover new authors — quality-gate before queuing
                    candidates = [
                        u
                        for u in _extract_annotators_from_pgn(pgn)
                        if u.lower() not in visited and len(visited) < max_users
                    ]
                    if candidates:
                        qualified = await _filter_users_by_quality(client, candidates)
                        for new_user in qualified:
                            key = new_user.lower()
                            if key not in visited and len(visited) < max_users:
                                visited.add(key)
                                queue.append(new_user)
    finally:
        if crawled_users_file:
            crawled_users_file.close()

    logger.info(
        "Crawl complete: %d users visited, %d with studies, %d still queued",
        len(visited),
        with_studies,
        len(queue),
    )


# Perf types to pull top players from (all guaranteed 2000+, skip quality gate)
_TOP_PLAYER_PERFS = ["classical", "rapid", "blitz"]


async def _fetch_top_players(
    client: httpx.AsyncClient,
    nb: int = 200,
) -> list[str]:
    """Return usernames of Lichess's top-nb players across classical/rapid/blitz."""
    usernames: set[str] = set()
    for perf in _TOP_PLAYER_PERFS:
        try:
            resp = await client.get(
                f"https://lichess.org/api/player/top/{nb}/{perf}",
                headers={"Accept": "application/json"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                for u in data.get("users", []):
                    usernames.add(u["username"])
                logger.info("  top/%s: %d players", perf, len(data.get("users", [])))
        except Exception as e:
            logger.warning("top players %s: %s", perf, e)
    logger.info("Top players total: %d unique", len(usernames))
    return list(usernames)


async def _fetch_team_members(
    client: httpx.AsyncClient,
    team_ids: list[str],
) -> list[str]:
    """Return usernames of members of the given Lichess teams (NDJSON API).

    Non-existent or private teams return 404/401 and are silently skipped.
    """
    usernames: list[str] = []
    for team_id in team_ids:
        try:
            resp = await client.get(
                f"https://lichess.org/api/team/{team_id}/users",
                headers={"Accept": "application/x-ndjson"},
                timeout=60.0,
            )
            if resp.status_code != 200:
                logger.debug("team/%s: status %d (skipping)", team_id, resp.status_code)
                continue
            count = 0
            for line in resp.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    u = json.loads(line)
                    name = u.get("username") or u.get("id", "")
                    if name:
                        usernames.append(name)
                        count += 1
                except Exception:
                    pass
            logger.info("team/%s: %d members", team_id, count)
        except Exception as e:
            logger.warning("team/%s: %s", team_id, e)
    return usernames


async def _discover_from_study_topics(
    client: httpx.AsyncClient,
    topics: list[str],
    max_studies_per_topic: int = 15,
) -> list[str]:
    """Extract annotator usernames from popular Lichess study topic pages.

    Lichess topic pages are JS-rendered, so only study IDs appear in the HTML.
    We extract those IDs and fetch each study's PGN to pull annotator names.
    """
    _study_id_re = re.compile(r"/study/([A-Za-z0-9]{8})")
    usernames: set[str] = set()

    for topic in topics:
        try:
            resp = await client.get(f"https://lichess.org/study/topic/{topic}/hot", timeout=30.0)
            if resp.status_code != 200:
                logger.debug("topic/%s: status %d", topic, resp.status_code)
                continue
            study_ids = list(dict.fromkeys(_study_id_re.findall(resp.text)))[:max_studies_per_topic]
            logger.info("topic/%s: %d study IDs found", topic, len(study_ids))
        except Exception as e:
            logger.warning("topic/%s: %s", topic, e)
            continue

        for study_id in study_ids:
            try:
                pgn_resp = await client.get(
                    f"https://lichess.org/study/{study_id}.pgn",
                    timeout=60.0,
                )
                if pgn_resp.status_code == 200:
                    found = _extract_annotators_from_pgn(pgn_resp.text)
                    usernames.update(found)
            except Exception:
                pass

    logger.info("Study topics total: %d unique annotators", len(usernames))
    return list(usernames)


async def discover_seed_users(
    client: httpx.AsyncClient,
    base_seed: list[str],
    cache_path: Path | None = None,
) -> list[str]:
    """Build an expanded seed list from multiple discovery sources:
      - Base hardcoded seed list
      - Lichess top-200 players per time control (2000+ rating guaranteed)
      - Lichess team members for chess education teams
      - Study authors from popular Lichess study topic pages

    All discovered users (except top players) pass through the quality gate.
    Results are cached to cache_path to avoid duplicate API calls on restart.
    """
    # Load from cache if available — skip all network discovery
    if cache_path and cache_path.exists():
        cached = [line.strip() for line in cache_path.read_text().splitlines() if line.strip()]
        logger.info(
            "Loaded %d seed users from cache %s (skipping discovery)", len(cached), cache_path
        )
        return cached

    seen: set[str] = {u.lower() for u in base_seed}
    result: list[str] = list(base_seed)

    def _add(users: list[str]) -> int:
        n = 0
        for u in users:
            if u.lower() not in seen:
                seen.add(u.lower())
                result.append(u)
                n += 1
        return n

    # Top players bypass quality gate (guaranteed 2000+)
    logger.info("Discovering top players...")
    top = await _fetch_top_players(client)
    n = _add(top)
    logger.info("Top players: +%d (total %d)", n, len(result))

    # Team members — quality gate applied below
    logger.info("Discovering team members (%d teams)...", len(LICHESS_STUDY_TEAMS))
    team_users = await _fetch_team_members(client, LICHESS_STUDY_TEAMS)
    new_team = [u for u in team_users if u.lower() not in seen]
    if new_team:
        qualified = await _filter_users_by_quality(client, new_team)
        n = _add(qualified)
        logger.info("Team members: +%d qualified (total %d)", n, len(result))

    # Study topic pages — quality gate applied below
    logger.info("Discovering study topic authors (%d topics)...", len(LICHESS_STUDY_TOPICS))
    topic_users = await _discover_from_study_topics(client, LICHESS_STUDY_TOPICS)
    new_topic = [u for u in topic_users if u.lower() not in seen]
    if new_topic:
        qualified = await _filter_users_by_quality(client, new_topic)
        n = _add(qualified)
        logger.info("Study topics: +%d qualified (total %d)", n, len(result))

    logger.info("Total seed users: %d", len(result))

    # Persist so restarts skip discovery entirely
    if cache_path:
        cache_path.write_text("\n".join(result) + "\n")
        logger.info("Seed cache written to %s", cache_path)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def build(
    output_path: Path,
    lichess_workers: int = 8,
    lichess_token: str = "",
    max_crawl_users: int = 2000,
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

    # ---- Phase 1b: Lichess BFS crawl (requires API token) ----
    if lichess_token:
        headers = {
            "Authorization": f"Bearer {lichess_token}",
            "User-Agent": "chess-ai-tutor/1.0 (educational; github.com/chess-ai)",
        }
        seed_cache_path = output_path.with_suffix(".seed_users")
        async with httpx.AsyncClient(
            headers=headers, timeout=180.0, follow_redirects=True
        ) as discovery_client:
            seed_users = await discover_seed_users(
                discovery_client, LICHESS_STUDY_AUTHORS, cache_path=seed_cache_path
            )

        crawled_users_path = output_path.with_suffix(".crawled_users")
        logger.info(
            "=== Phase 1b: Lichess BFS crawl (seed=%d accounts, max=%d) ===",
            len(seed_users),
            max_crawl_users,
        )
        async for label, pgn_text in crawl_lichess_studies(
            token=lichess_token,
            seed_users=seed_users,
            workers=lichess_workers,
            max_users=max_crawl_users,
            crawled_users_path=crawled_users_path,
        ):
            positions = list(_parse_pgn_annotations(pgn_text, source=label))
            n = write_positions(positions)
            if n:
                logger.info("  %s → %d new positions (total: %d)", label, n, total_written)
                total_written += n
        logger.info("After BFS crawl: %d total positions", total_written)
    else:
        logger.info("=== Phase 1b: Skipped (no --lichess-token provided) ===")

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
    parser.add_argument("--workers", type=int, default=8, help="Concurrent HTTP workers")
    parser.add_argument(
        "--lichess-token",
        type=str,
        default="",
        help="Lichess personal API token for authenticated study crawl",
    )
    parser.add_argument(
        "--max-crawl-users",
        type=int,
        default=2000,
        help="Maximum number of Lichess users to crawl (BFS limit)",
    )
    args = parser.parse_args()

    asyncio.run(
        build(
            output_path=args.output,
            lichess_workers=args.workers,
            lichess_token=args.lichess_token,
            max_crawl_users=args.max_crawl_users,
        )
    )


if __name__ == "__main__":
    main()
