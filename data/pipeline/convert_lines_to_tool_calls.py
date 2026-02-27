"""Convert lines_30k.jsonl into Qwen3 tool-call SFT format.

Each record becomes a 5-turn conversation:
  system  → line-generator system prompt + stockfish_analyze tool def
  user    → FEN + ASCII board + move played
  assistant (tool call) → stockfish_analyze(fen, depth=15, multipv=3)
  tool    → synthetic Stockfish PV output (reconstructed from stored lines)
  assistant (final)     → formatted LINE 1/2/3 output

Usage:
    uv run python data/pipeline/convert_lines_to_tool_calls.py \\
        --input  data/processed/lines_30k.jsonl \\
        --output data/processed/lines_tool_sft.jsonl \\
        --eval-split 0.05 \\
        --eval-output data/processed/lines_tool_sft_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import chess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a chess line generator. Given a FEN position and the move just played, "
    "call stockfish_analyze to get the top engine continuations, then format them as "
    "structured lines with inline move annotations.\n\n"
    "Output format:\n"
    "LINE 1: move (purpose) → move (purpose) → ... | eval: <label>\n"
    "LINE 2: ...\n"
    "LINE 3: ...\n\n"
    "Eval labels (from White's perspective): "
    "winning for white | good for white | equal | good for black | winning for black"
)

TOOL_DEF = [
    {
        "type": "function",
        "function": {
            "name": "stockfish_analyze",
            "description": (
                "Analyze a chess position with Stockfish. "
                "Returns the top N principal variations with centipawn evaluations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string",
                        "description": "FEN string of the position to analyze",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Search depth (default 15)",
                    },
                    "multipv": {
                        "type": "integer",
                        "description": "Number of lines to return (default 3)",
                    },
                },
                "required": ["fen"],
            },
        },
    }
]

# Coarse centipawn estimate per eval label (white perspective).
# Used only to produce a plausible synthetic tool result — not shown in training targets.
_LABEL_TO_CP: dict[str, int] = {
    "winning for white": 450,
    "good for white": 200,
    "equal": 0,
    "good for black": -200,
    "winning for black": -450,
}

_STRIP_ANNOTATION = re.compile(r"\([^)]*\)")


def _bare_moves(line_str: str) -> str:
    """Strip annotations and eval suffix, return bare UCI-like SAN move sequence."""
    # Remove "LINE N: " prefix
    body = re.sub(r"^LINE\s+\d+\s*:\s*", "", line_str)
    # Remove "| eval: ..." suffix
    body = body.split("| eval:")[0].strip()
    # Remove parenthetical annotations
    body = _STRIP_ANNOTATION.sub("", body)
    # Replace → separators with spaces and normalise whitespace
    return " ".join(body.replace("→", " ").split())


def _label_from_line(line_str: str) -> str:
    if "| eval:" in line_str:
        return line_str.split("| eval:")[-1].strip()
    return "equal"


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------


def _board_ascii(fen: str) -> str:
    try:
        board = chess.Board(fen)
        rows = str(board).split("\n")
        lines = ["  a b c d e f g h"]
        for i, row in enumerate(rows):
            lines.append(f"{8 - i} {row}")
        return "\n".join(lines)
    except Exception:
        return "(board unavailable)"


def _build_user_msg(sample: dict) -> str:
    fen = sample["fen"]
    move = sample["move_san"]
    board_str = _board_ascii(fen)
    return (
        f"Position (FEN): {fen}\n\n"
        f"Board:\n{board_str}\n\n"
        f"Move played: {move}\n\n"
        "Generate the 3 key engine lines for this position so I can coach the student."
    )


def _build_tool_result(sample: dict) -> str:
    """Reconstruct a plausible Stockfish tool result from the stored lines.

    The raw Stockfish PVs were not saved, so we reconstruct them from the
    formatted lines. This gives the model a realistic tool result to learn
    from — the cp values are coarse band estimates, which is fine for SFT.
    """
    parts: list[str] = []
    for i, line_str in enumerate(sample["lines"], 1):
        label = _label_from_line(line_str)
        cp = _LABEL_TO_CP.get(label.lower(), 0)
        moves = _bare_moves(line_str)
        sign = "+" if cp >= 0 else ""
        parts.append(f"PV{i} (depth 15): {moves} | score cp {sign}{cp}")
    return "\n".join(parts)


def _build_assistant_final(sample: dict) -> str:
    return "\n".join(sample["lines"])


def _build_tool_call_args(sample: dict) -> str:
    return json.dumps({"fen": sample["fen"], "depth": 15, "multipv": 3})


def sample_to_messages(sample: dict) -> list[dict]:
    """Convert a lines_30k record into the 5-turn message list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_msg(sample)},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_0",
                    "function": {
                        "name": "stockfish_analyze",
                        "arguments": _build_tool_call_args(sample),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": _build_tool_result(sample),
            "tool_call_id": "call_0",
        },
        {"role": "assistant", "content": _build_assistant_final(sample)},
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
        default="data/processed/lines_tool_sft.jsonl",
        help="Output JSONL (train split)",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.05,
        help="Fraction of data reserved for eval (default 0.05)",
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed/lines_tool_sft_eval.jsonl",
        help="Output JSONL (eval split)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────────────────
    src = Path(args.input)
    if not src.exists():
        log.error("Input file not found: %s", src)
        sys.exit(1)

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

    log.info("Loaded %d valid samples (%d skipped)", len(all_samples), skipped)

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
                    "tools": TOOL_DEF,
                    "metadata": {
                        "source": "lichess_lines_tool_sft",
                        "fen": s["fen"],
                        "move_san": s["move_san"],
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Wrote %d records to %s", len(samples), out_path)

    write_split(train_samples, args.output)
    write_split(eval_samples, args.eval_output)

    log.info("Done.")


if __name__ == "__main__":
    main()
