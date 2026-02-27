"""Convert lines_30k.jsonl into the line-generator SFT format.

Format (3-turn conversation, no tool calls):

  system    → LINE_GENERATOR_SYSTEM_PROMPT
  user      → board ascii + FEN + "Move played: X" + task instruction
  assistant → (think block stripped from loss) + <line>LINE 1...</line> × 3

The <think> block in the assistant target is left empty — the model will
develop its own reasoning during GRPO.  strip_think_from_target() in
training/lib.py removes it from the SFT loss so the model only gets signal
on the <line> outputs.

Usage:
    uv run python data/pipeline/convert_lines_to_sft.py \\
        --input  data/processed/lines_30k.jsonl \\
        --output data/processed/lines_sft.jsonl \\
        --eval-split 0.05 \\
        --eval-output data/processed/lines_sft_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import chess

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import (
    LINE_GENERATOR_SYSTEM_PROMPT,
    board_ascii,
    format_line_generator_prompt,
    move_facts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _consensus_eval(lines: list[str]) -> str:
    """Pick the most common eval label across the stored lines."""
    from collections import Counter

    labels: list[str] = []
    for line in lines:
        if "| eval:" in line:
            labels.append(line.split("| eval:")[-1].strip().lower())
    if not labels:
        return "equal"
    return Counter(labels).most_common(1)[0][0]


def _wrap_lines(raw_lines: list[str]) -> str:
    """Wrap each LINE string in <line>...</line> tags."""
    return "\n".join(f"<line>{l.strip()}</line>" for l in raw_lines)


def sample_to_messages(sample: dict) -> list[dict]:
    """Convert a lines_30k record into the 3-turn SFT message list."""
    fen = sample["fen"]
    move_uci = sample.get("move_uci", "")
    move_san = sample["move_san"]

    # Board ascii with turn indicator (reuses the coach helper)
    try:
        board = chess.Board(fen)
        board_str = board_ascii(board)
    except Exception:
        board_str = "(board unavailable)"
        board = None

    # Derive verified move facts from python-chess (free, deterministic)
    facts: list[str] = []
    if board is not None and move_uci:
        try:
            move = chess.Move.from_uci(move_uci)
            facts = move_facts(board, move)
        except Exception:
            pass

    # Eval string from consensus of stored lines (no cp numbers shown)
    eval_str = _consensus_eval(sample["lines"])

    user_content = format_line_generator_prompt(board_str, fen, move_san, eval_str, facts)

    # Assistant target: empty <think> block (stripped by lib.py) + <line> outputs
    # The think block is intentionally empty — GRPO will teach the model what
    # to think; SFT only teaches the output format.
    assistant_content = "<think>\n</think>\n" + _wrap_lines(sample["lines"])

    return [
        {"role": "system", "content": LINE_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/lines_30k.jsonl",
        help="Source JSONL with lichess_lines records",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lines_sft.jsonl",
        help="Output JSONL (train split)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.05,
        help="Fraction reserved for eval (default 0.05)",
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed/lines_sft_eval.jsonl",
        help="Output JSONL (eval split)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        log.error("Input not found: %s", src)
        sys.exit(1)

    # ── load ──────────────────────────────────────────────────────────────
    all_samples: list[dict] = []
    skipped = 0
    with src.open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                if rec.get("metadata", {}).get("source") == "lichess_lines" and rec.get("lines"):
                    all_samples.append(rec)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1

    log.info("Loaded %d samples (%d skipped)", len(all_samples), skipped)

    # ── split ─────────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_eval = max(1, int(len(all_samples) * args.eval_split))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    log.info("Split: %d train / %d eval", len(train_samples), len(eval_samples))

    # ── convert & write ───────────────────────────────────────────────────
    def write_split(samples: list[dict], out_path: str) -> None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for s in samples:
                msgs = sample_to_messages(s)
                record = {
                    "messages": msgs,
                    "metadata": {
                        "source": "lichess_lines_sft",
                        "fen": s["fen"],
                        "move_san": s["move_san"],
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Wrote %d records → %s", len(samples), out_path)

    write_split(train_samples, args.output)
    write_split(eval_samples, args.eval_output)
    log.info("Done.")


if __name__ == "__main__":
    main()
