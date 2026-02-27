# RL Training Plan — Chess Tutor

## Goals (priority order)

1. **Correctness** — all chess facts, legal move sequences, accurate evaluations
2. **Tone** — treats the move as coming from a human student; encouraging not cold
3. **Educational value** — explains *why*, not just *what*; names patterns correctly
4. **Conciseness** — no padding, no repetition, appropriate length to move severity

---

## Architecture: Two-Stage Pipeline

Rather than trying to verify claims embedded in free-form prose, split the task cleanly:

```
Stage 1 — Line Generator (verifiable, GRPO)
  Input:  FEN position + move played
  Output: N key lines in structured format, each ending with an evaluation label

Stage 2 — Coach (soft quality, Haiku judge or SFT only)
  Input:  FEN + move + verified key lines
  Output: natural language coaching comment
```

**Why this is better than embedding structured blocks in coaching output:**
- Line generation is a well-defined subtask with deterministic ground truth (Stockfish)
- The coach never needs to assert chess facts — it just narrates verified lines
- Each stage can be trained, evaluated, and improved independently
- Stage 1 correctness is fully verifiable; Stage 2 quality is judged on tone/clarity only

---

## Stage 1: Line Generator

### Output Format

Each move is annotated with its purpose in parentheses — making lines educational rather
than just a sequence of moves. The eval label covers the final position only.

```
LINE 1: e4 (control center, gain space) → Nf6 (challenge center, develop knight) → Nc3 (reinforce center, develop) → d5 (counterattack) | eval: equal
LINE 2: e4 (control center) → d5 (contest center immediately) → exd5 (capture, open e-file) → Qxd5 (recapture but loses tempo) | eval: good for white
LINE 3: e4 (control center) → Nf6 (develop, pressure e4) → e5 (kick knight, gain space) → Nd5 (retreat to strong square) → Nf3 (develop, attack knight) | eval: very good for white
```

Fields:
- `LINE N:` — moves with inline purpose annotations: `move (purpose)`
- `→` — move separator
- `eval:` — one of:
  `winning for white`, `good for white`,
  `equal`,
  `good for black`, `winning for black`

**What is verifiable vs judged:**
- Move sequence + eval label → verified by python-chess + Stockfish (free, deterministic)
- Move purpose annotations → judged by GRPO reward R4 (Haiku, cheap)

### Verification with Stockfish (GRPO reward signals)

**R1 — Move Legality (free, deterministic)**
Play each line with python-chess from the given FEN.
- +1.0 per line if fully legal
- -1.0 per line if any move is illegal (hard penalty — illegal moves are never acceptable)

**R2 — Opponent Move Quality (free, Stockfish)**
At each opponent move in a line, query Stockfish for the best move at depth 12.
Penalise if opponent plays a move that loses >150cp vs Stockfish best.
- -0.5 per opponent move that is a blunder (prevents "assume opponent blunders" lines)
- 0.0 if opponent move is within 150cp of best (reasonable play)

**R3 — Final Position Evaluation Accuracy (free, Stockfish)**
Run Stockfish at depth 18 on the final position of each line.
Map cp score to evaluation label:

| Stockfish cp (white perspective) | Label |
|---|---|
| > +300 | winning for white |
| +100 to +300 | good for white |
| -100 to +100 | equal |
| -100 to -300 | good for black |
| < -300 | winning for black |

- +1.0 if model label matches Stockfish label
- -0.5 if off by one band
- -1.0 if completely wrong (e.g. "equal" when Stockfish says "winning")

**R4 — Move Annotation Quality (optional, Phase 2, Haiku judge)**
Each move has a purpose annotation in parentheses. Haiku judges a sample of annotations per line:
- Are the annotations accurate chess concepts (not vague filler like "good move")?
- Do they reflect the actual purpose of the move in context (tactics, strategy, development)?
- Are they concise (3-6 words per annotation)?

Score: 0.0–1.0 normalised from Haiku rubric. Cost: ~$0.001/line → negligible.
*Skip in Phase 1 — implement after verifiable rewards (R1/R2/R3/R5) are validated.*

**R5 — Line Relevance (free, Stockfish)**
A line is relevant if it illustrates the consequence of the move played.
Proxy: the first move of the line must be the move played (or the best response to it).
Lines that ignore the played move and analyse a random position get 0.

**Combined Stage 1 Reward:**
```python
R = 0.25 * R1 + 0.20 * R2 + 0.30 * R3 + 0.15 * R4 + 0.10 * R5
```

Legality (R1) is a hard gate — if 0, the line is useless regardless of other scores.
Eval accuracy (R3) is highest weight — it's the most informative correctness signal.
Move annotations (R4) are rewarded but not dominant — they enrich lines, not define them.

### Bootstrapping the `<think>` Block

No special handling needed. The SFT data format is:

```
User:  [FEN + move, no eval numbers shown]
Assistant: <think>[stripped by strip_think_from_target()]</think>
LINE 1: Nf6 (challenge center) → Nc3 (reinforce center) → d5 (counterattack) | eval: equal
LINE 2: d5 (contest center) → exd5 (capture) → Qxd5 (recapture, tempo loss) | eval: good for white
LINE 3: Nf6 (develop) → e5 (kick knight) → Nd5 (strong square) | eval: very good for white
```

**Generating move annotations for SFT data**: Stockfish gives moves + evals for free.
For the purpose annotations, use the fine-tuned chess-tutor model (after SFT completes)
to annotate each move in a line — self-distillation, runs locally, zero API cost.
Alternatively use Haiku during data generation (~$0.001/line, trivial cost).
A template fallback (captures → "recapture", checks → "give check") covers edge cases.

`strip_think_from_target()` in `training/lib.py` already removes `<think>` from SFT loss
targets — the model is never penalized for *what* it thinks, only for the structured output.
During inference it produces whatever thinking it wants. GRPO later rewards thinking that
leads to correct lines.

**Do NOT show cp scores in the user prompt** — the model must output eval labels
(`equal`, `good for white`, etc.) based on its own reasoning, not by copying numbers.
Stockfish cp is used only to generate the ground-truth labels for SFT targets.

### Training Data for Stage 1 SFT

**Do NOT reuse existing `train.jsonl` positions** — the coaching model has already seen
them. Use fresh positions from Lichess via HuggingFace.

**Dataset**: `austindavis/lichess_uci`
- 5.57B games with ELO ratings for both players
- Sample strategy: mix amateur (1200-1600 ELO) + intermediate (1600-2000) + strong (2000+)
- Target: ~50k positions (easily enough, fast to generate)

**Pipeline** (`data/pipeline/generate_lines.py`):
1. Load Lichess games from HuggingFace, sample by ELO tier
2. Replay each game with python-chess to extract (FEN, move_san) at interesting positions
   - Skip opening moves 1-8 (too theory-dependent)
   - Prefer positions where Stockfish eval changes significantly (mistakes/blunders)
3. Run Stockfish `multipv 3` at depth 15 from each position
   - Each PV is a line; play it out to get the final FEN
   - Run Stockfish again on final position to get cp eval
   - Map cp → eval label (no cp shown in training data):

| Stockfish cp (white perspective) | Label |
|---|---|
| > +300 | winning for white |
| +100 to +300 | good for white |
| -100 to +100 | equal |
| -100 to -300 | good for black |
| < -300 | winning for black |

4. Format as `LINE N: <SAN moves> \| eval: <label>` — no cp numbers anywhere
5. Output JSONL with `{"fen": ..., "move": ..., "lines": [...], "metadata": {"source": "lichess_lines"}}`

**Scale**: 50k positions × 3 lines = 150k line verification examples for SFT + GRPO.

---

## Stage 2: Coach

Takes verified lines as additional context in the prompt. Since lines are now verified,
the coach never needs to assert chess facts — it only needs to:
- Interpret what the lines mean for the student
- Frame it with the right tone (encouraging, educational)
- Be appropriately concise

**Training**: SFT on existing textbook data is likely sufficient for Stage 2. The coach
already learns tone and style from imitation. GRPO for Stage 2 is optional — only needed
if tone/conciseness is still poor after SFT.

If Stage 2 GRPO is needed:
- Single reward: Haiku judge scoring tone + educational value + conciseness
- ~$5 per 5k rollouts at Haiku pricing

---

## Implementation Order

### Phase 1 — Stage 1 Data + SFT (prerequisite for GRPO)
- [ ] `data/pipeline/generate_lines.py` — pull Lichess games from `austindavis/lichess_uci`,
  sample by ELO tier, extract (FEN, move) at non-opening positions, run Stockfish `multipv 3`,
  map cp → eval label (no cp in output), emit JSONL
- [ ] Add line generator system prompt to `src/tutor/prompts.py`
- [ ] `recipes-train/qwen3-4b-lines/` — SFT recipe for line generator task
  (reuses `training/lib.py`, new `train.py` with same 8-bit DDP setup)

### Phase 2 — Stage 1 GRPO
- [ ] `src/verification/rewards.py` — R1 legality, R2 opponent quality, R3 eval accuracy, R4 relevance
- [ ] `training/lib_rl.py` — GRPO loop: rollout generation, reward aggregation, policy update
- [ ] `recipes-train/qwen3-4b-grpo/` — GRPO recipe wrapping lib_rl

### Phase 3 — Integration
- [ ] Update `src/tutor/prompts.py` — pipe Stage 1 output into Stage 2 prompt
- [ ] Update `src/tutor/web.py` / `src/tutor/review.py` — two-stage inference call
- [ ] Eval: compare single-stage SFT vs two-stage (SFT lines + SFT coach) on 50 real games

### Phase 4 — Stage 2 GRPO (optional, only if needed)
- [ ] `src/tutor/judge.py` — async Haiku judge with structured rubric
- [ ] GRPO run on coach model with Haiku reward

---

## Open Questions

- **Single model vs two models**: can one fine-tuned 4B model do both stages (different
  system prompts)? Likely yes for POC — avoids loading two models.
- **K lines**: N=3 lines balances coverage vs token budget. N=5 if context allows.
- **GRPO library**: TRL's `GRPOTrainer` vs custom loop. TRL is simpler; custom loop
  needed if async Stockfish queries between steps are required.
- **When to stop GRPO**: track R3 (eval accuracy) on held-out set; stop when it plateaus.
