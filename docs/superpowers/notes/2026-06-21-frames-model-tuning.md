# FRAMES hill-climbing notes (2026-06-21)

Iterative tuning of the Wikipedia agent against the FRAMES benchmark. Method:
fixed 20-sample subset (`--limit 20`) for comparability, **one change per cycle**
to isolate impact, Haiku judge held constant (`correctness_judge` =
`anthropic/claude-haiku-4-5`). Metric = `model_graded_qa` accuracy on 20 samples
(stderr ≈ 0.10, so single-cycle deltas are directional, not significant — confirm
winners on the full 100 / 824 before treating as final).

Analysis helper: `eval/scripts/analyze_run.py` (per-sample correctness, grounding
recall, gold-vs-read pages, step-cap count).

## Round 1 — harness & prompt (Haiku agent)

| Stage | Change | Correctness | Recall |
|------|--------|------------|--------|
| Baseline | `max_steps=6` | 0.300 | 0.375 |
| Cycle 1 | `max_steps` 6→20 | 0.300 | 0.455 |
| Cycle 2 | decisive prompt rewrite | 0.450 | 0.444 |
| Cycle 3 | forced final answer on budget exhaustion | 0.450 | 0.492 |
| Cycle 4 | fetch article **body** (drop `exintro`, 1500→4000 chars) | **0.550** | 0.491 |

Biggest movers: **decisive prompt (+0.15)** and **body extracts (+0.10)**.
`max_steps` alone lifted recall but not correctness (enabler, not sufficient).
Forced-answer was neutral on this set (kept — removes guaranteed-wrong canned
non-answers). All committed.

## Round 2 — model axis (anchor = Round 1 Haiku best, 0.550)

| Stage | Isolated change | Correctness | Wall-clock |
|------|--------|------------|-----------|
| **M1** | Haiku → **Sonnet 4.6**, no thinking | **0.750** | 1:34 |
| M2 | + adaptive thinking (effort high) | 0.650 | 3:11 |
| M2b | adaptive thinking, effort=low | 0.700 | 3:05 |
| M3 | + chain-of-thought in prompt | 0.700 | 1:43 |
| M4 | + structured `ANSWER:` output format | 0.750 | 1:41 |

**Only the Sonnet upgrade improved the score (+0.20, +36% relative).** The other
three were neutral-to-negative:

- **Thinking budget** — adaptive thinking *regressed* correctness at every effort
  tried and ~2×'d latency. The multi-step tool loop already supplies the reasoning
  scaffold; extended thinking is redundant here. **Off by default.**
- **Chain-of-thought prompt** — neutral/slightly negative. Sonnet reasons
  adequately without being told to.
- **Output format (`ANSWER:`)** — neutral (Haiku judge already extracted terse
  answers reliably) and would clutter the CLI UX. Reverted.

## Committed outcome

- Default `AGENT_MODEL = claude-sonnet-4-6` (the validated win).
- `THINKING` / `EFFORT` plumbing added to `config.py` + `agent.py`, gated and
  `None` by default (Haiku-safe; one-line opt-in for Sonnet experiments).
- Prompt and tool settings unchanged from Round 1.

## Next levers (not yet done)

- Confirm the +0.20 on the full 100/824 (current numbers are n=20).
- Remaining Sonnet failures cluster in: numeric/temporal arithmetic, and
  search-quality misses (tangential pages). Try `DEFAULT_SEARCH_LIMIT`↑ or query
  reformulation for the latter.
- Consider a stronger judge (Sonnet) — current judge is Haiku; validate it isn't
  the ceiling.
