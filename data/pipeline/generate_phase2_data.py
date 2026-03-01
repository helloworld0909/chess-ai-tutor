"""Generate Phase 2 Joint Task training data from textbook positions and coaching cache.

Usage:
    uv run python data/pipeline/generate_phase2_data.py \
        --train-data data/processed/train.jsonl \
        --coaching-cache data/processed/.llm_coaching_cache.jsonl \
        --output data/processed/lines_joint_sft.jsonl \
        --eval-output data/processed/lines_joint_sft_eval.jsonl \
        --eval-split 0.05 \
        --workers 8 \
        --depth 18
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from tutor.prompts import (
    JOINT_SYSTEM_PROMPT,
    board_ascii,
    format_joint_user_prompt,
    move_facts,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.pipeline.convert_lines_to_sft_thinking import (
    _PIECE_NAMES,
    _move_purpose,
    _get_engine,
    _game_phase,
    _cp_to_label,
    _build_thinking,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _analyze_full_lines(fen: str, depth: int, multipv: int = 5):
    engine = _get_engine()
    board = chess.Board(fen)
    try:
        result = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    except Exception:
        return None, [], []

    candidates = []
    root_cp = None
    full_lines = []

    for i, info in enumerate(result):
        pv = info.get("pv", [])
        if not pv:
            continue
            
        score = info.get("score")
        cp = None
        if score is not None:
            if score.is_mate():
                if i == 0: root_cp = None
            else:
                cp_val = score.white().score()
                if i == 0: root_cp = cp_val
                cp = cp_val
                
        # Candidate array for thinking block
        first_move = pv[0]
        try:
            cand_san = board.san(first_move)
            cand_purpose = _move_purpose(board, first_move)
            candidates.append((cand_san, cand_purpose, cp))
        except Exception:
            continue
        
        # Build full line
        line_board = board.copy()
        current_line = []
        for mv in pv:
            try:
                san = line_board.san(mv)
                purpose = _move_purpose(line_board, mv)
                current_line.append((san, purpose))
                line_board.push(mv)
            except Exception:
                break
                
        eval_label = "equal"
        if cp is not None:
            if cp >= 200: eval_label = "winning for white"
            elif cp >= 60: eval_label = "good for white"
            elif cp >= -60: eval_label = "equal"
            elif cp >= -200: eval_label = "good for black"
            else: eval_label = "winning for black"
        elif score and score.is_mate():
            if score.white().mate() > 0:
                eval_label = "winning for white"
            else:
                eval_label = "winning for black"
                
        full_lines.append((current_line, eval_label))

    return root_cp, candidates, full_lines


def convert_sample(args_tuple: tuple) -> dict | None:
    sample, comment, depth, seed_offset = args_tuple
    metadata = sample.get("metadata", {})
    fen = metadata.get("fen")
    move_uci = metadata.get("move_uci")
    if not fen or not move_uci:
        return None
        
    rng = random.Random(hash(fen + move_uci) + seed_offset)

    try:
        board = chess.Board(fen)
        board_str = board_ascii(board)
        move = chess.Move.from_uci(move_uci)
        move_san = board.san(move)
    except Exception:
        return None

    facts = move_facts(board, move)
    board_after = board.copy()
    try:
        board_after.push(move)
    except Exception:
        return None
    board_after_str = board_ascii(board_after)
    fen_after = board_after.fen()

    root_cp, candidates, full_lines = _analyze_full_lines(fen, depth=depth, multipv=5)
    
    if not candidates or not full_lines:
        return None
    
    eval_str = "a forced mate sequence exists" if root_cp is None else _cp_to_label(root_cp, board.turn)
    
    key_lines_for_prompt = []
    assistant_lines = []
    
    for idx, (moves_info, eval_label) in enumerate(full_lines[:5]):
        sans_only = " → ".join(m for m, p in moves_info[:10]) 
        key_lines_for_prompt.append(sans_only)
        
        annotated_moves = " → ".join(f"{m} ({p})" for m, p in moves_info[:10])
        assistant_lines.append(f"<line>LINE {idx+1}: {annotated_moves} | eval: {eval_label}</line>")

    user_content = format_joint_user_prompt(
        board_str, fen, move_san, eval_str, facts, board_after_str, fen_after, key_lines=key_lines_for_prompt
    )

    thinking = _build_thinking(fen, root_cp, candidates, rng)
    
    assistant_content = f"<think>\n{thinking}\n</think>\n"
    assistant_content += "\n".join(assistant_lines)
    assistant_content += f"\n\n{comment}"

    return {
        "messages": [
            {"role": "system", "content": JOINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "textbook_joint_sft",
            "fen": fen,
            "move_san": move_san,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="data/processed/train.jsonl")
    parser.add_argument("--coaching-cache", default="data/processed/.llm_coaching_cache.jsonl")
    parser.add_argument("--output", default="data/processed/lines_joint_sft.jsonl")
    parser.add_argument("--eval-output", default="data/processed/lines_joint_sft_eval.jsonl")
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train_data)
    cache_path = Path(args.coaching_cache)
    
    if not train_path.exists() or not cache_path.exists():
        log.error(f"Input not found. Train: {train_path.exists()}, Cache: {cache_path.exists()}")
        sys.exit(1)

    # 1. Load Coaching Cache
    cache = {}
    with cache_path.open() as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            key = rec.get("_key") or rec.get("key") or rec.get("cache_key")
            val = rec.get("coaching") or rec.get("result")
            if key and val:
                cache[key] = val

    all_samples = []
    with train_path.open() as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            
            metadata = rec.get("metadata", {})
            fen = metadata.get("fen")
            move_uci = metadata.get("move_uci")
            if not fen or not move_uci: continue
            
            key_str = f"llm9:textbook:{fen}:{move_uci}:True"
            k = hashlib.md5(key_str.encode("utf-8")).hexdigest()
            
            comment = cache.get(k)
            if not comment:
                k2 = hashlib.md5(f"llm9:textbook:{fen}:{move_uci}:False".encode("utf-8")).hexdigest()
                comment = cache.get(k2)
                
            if isinstance(comment, dict) and comment.get("_skip"):
                continue
                
            if comment and isinstance(comment, str):
                if "<comment>SKIP</comment>" in comment or comment.strip() == "SKIP":
                    continue
                if "<comment>" in comment:
                    c_start = comment.find("<comment>") + 9
                    c_end = comment.find("</comment>")
                    comment = comment[c_start:c_end].strip()
                    
                all_samples.append((rec, comment))

    log.info("Loaded %d textbook positions with coaching match", len(all_samples))

    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_eval = max(1, int(len(all_samples) * args.eval_split))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    
    log.info("Split: %d train / %d eval", len(train_samples), len(eval_samples))

    train_tagged = [(s, c, args.depth, args.seed, "train") for s, c in train_samples]
    eval_tagged = [(s, c, args.depth, args.seed, "eval") for s, c in eval_samples]
    all_tagged = train_tagged + eval_tagged

    train_out = Path(args.output)
    eval_out = Path(args.eval_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    
    written = {"train": 0, "eval": 0}
    failed = {"train": 0, "eval": 0}

    ctx = multiprocessing.get_context("spawn")
    with (
        train_out.open("w", encoding="utf-8") as f_train,
        eval_out.open("w", encoding="utf-8") as f_eval,
        ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool,
    ):
        futures = {
            pool.submit(convert_sample, (s, c, depth, seed)): split
            for s, c, depth, seed, split in all_tagged
        }

        for i, fut in enumerate(as_completed(futures)):
            split = futures[fut]
            try:
                result = fut.result()
                if result:
                    fout = f_train if split == "train" else f_eval
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    written[split] += 1
                else:
                    failed[split] += 1
            except Exception as e:
                failed[split] += 1

            if (i + 1) % 100 == 0:
                log.info("  %d / %d done (written: %d, failed: %d)", i + 1, len(all_tagged), sum(written.values()), sum(failed.values()))

    log.info("Done. train=%d eval=%d failed_train=%d failed_eval=%d", written["train"], written["eval"], failed["train"], failed["eval"])

if __name__ == "__main__":
    main()
