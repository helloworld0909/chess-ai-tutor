# RL Training Ideas — Chess Tutor

Post-SFT reinforcement learning to improve coaching quality beyond what
imitation learning on 30k samples can achieve.

## Core Insight

Chess coaching has two reward regimes:
- **Verifiable** — Stockfish and python-chess can fact-check specific claims
- **Soft** — pedagogical clarity requires a judge model

The Stockfish facts are already in the user prompt, so naive verifiable
rewards (e.g. "did the model mention the best move?") are trivially solvable
by reading the prompt. Useful RL signal must come from claims *not* in the
prompt.

---

## Reward Signals

### 1. Line Verification (verifiable, not in prompt)
If the model asserts a concrete variation — e.g. "after Qxf7+ Kh8 Re8#" —
verify the line with python-chess:
- Parse all SAN sequences in the response
- Play them out from the position FEN
- Check legality and final board state / outcome
- Reward correct lines, penalise illegal or wrong-outcome lines

### 2. Tactic Pattern Detection (verifiable, not in prompt)
Use python-chess to independently detect tactical motifs in the position:
- fork, pin, skewer, discovered attack, back-rank weakness, overloaded piece
- If a tactic exists, reward the model for correctly naming it
- Penalise false tactic claims ("this is a fork" when no fork exists)

### 3. Counterfactual Verification (verifiable, cheap Stockfish query)
If the model suggests an alternative — e.g. "Kh1 would have been safe" —
run a one-off Stockfish query on that position to verify the evaluation
claim. Not in the original prompt, so not trivially copyable.

### 4. Judge LLM Quality Score (soft)
Use Claude Haiku or GPT-4o-mini API to rate:
- Specificity (concrete vs generic)
- Explanation depth appropriate to move severity
  (blunders warrant more than inaccuracies)
- Absence of hallucinated chess claims

---

## Hardware Workaround (2× RTX 5090)

Running judge + trainee simultaneously doesn't fit in 64 GB.
Recommended approach: **API judge + iterative offline DPO**.

```
Loop N iterations:
  1. Load trainee → generate K completions per position → save to disk
  2. Unload trainee
  3. Score with Claude Haiku API (no GPU) + python-chess verifier
  4. Build preference pairs (best vs worst completion per position)
  5. Load trainee → DPO update on preference pairs
  6. Repeat
```

This avoids online RL infrastructure complexity while still improving
iteratively. No simultaneous judge + trainee needed.

Alternative for verifiable-only rewards (line verification, tactic detection):
these need no judge GPU at all — run GRPO with purely algorithmic rewards.

---

## Algorithm Recommendations

| Phase | Algorithm | Reason |
|-------|-----------|--------|
| Initial RL | Iterative DPO | Simple, stable, no reference model divergence issues |
| Verifiable rewards only | GRPO | No judge needed, scales with more positions |
| Full mixed rewards | GRPO + API judge | API judge scores batched between gradient steps |

---

## Implementation Order

- [ ] Build line verification reward: parse SAN from response → python-chess legality check
- [ ] Build tactic detection reward: python-chess position analysis → motif labels
- [ ] Build preference pair generator: K rollouts per position, rank by reward
- [ ] Implement iterative DPO training loop
- [ ] Integrate API judge scoring (Claude Haiku) for soft reward signal
- [ ] Experiment with GRPO once verifiable rewards are validated
