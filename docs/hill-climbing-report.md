# Wiki Agent — Hill-Climbing Report

A running log of attempts to improve the agent's benchmark scores. Each "round"
is one focused tuning effort (often by a different agent). New efforts **append a
new round** — keep the older rounds intact so we don't lose the history of what
was tried and why.

If you only read one thing, read **Biggest wins** and **Current best config**
below.

---

## Biggest wins so far

Ranked by impact on FRAMES correctness (20-sample subset). Full detail in the
rounds below.

| Rank | Change | Δ correctness | Round |
|------|--------|--------------:|-------|
| 1 | Upgrade agent model Haiku → **Sonnet 4.6** | **+0.20** | R2 |
| 2 | **Decisive prompt** (decompose, commit to an answer, don't loop) | +0.15 | R1 |
| 3 | Fetch article **body** instead of intro-only (`exchars` 1500→4000) | +0.10 | R1 |
| — | `max_steps` 6→20 | +0.00 corr / +0.08 recall (enabler) | R1 |

**Things that did _not_ help** (tested and dropped): extended/adaptive thinking,
chain-of-thought prompting, structured `ANSWER:` output format. See R2.

---

## Current best config

The settings the agent ships with today (all in `agent/wiki_agent/`).

| Setting | Value | Where | Why |
|--------|-------|-------|-----|
| Agent model | `claude-haiku-4-5` (default) | `config.py` `AGENT_MODEL` | cheap/fast for dev |
| — quality lever | `claude-sonnet-4-6` on demand | `run(model=...)` | +0.20 correctness (R2) |
| Step budget | `max_steps=20` | `eval/.../tasks.py` `frames()` | FRAMES needs 2–15 article reads (R1) |
| Article fetch | body, `exchars=4000` | `wikipedia.get_article` | facts live below the lead (R1) |
| Budget-exhaustion | forced final answer | `agent.run` | no canned non-answers (R1) |
| System prompt | decisive v2 | `agent.SYSTEM_PROMPT` | stop looping, commit (R1) |

**Score on the 20-sample FRAMES subset:** Haiku default ≈ **0.55**; Sonnet ≈
**0.75**. Grounding recall ≈ 0.50. (Extended thinking was tested and is not used
— no gain, ~2× latency; see R2.)

---

## How we measure (conventions)

So rounds stay comparable, every round uses the same protocol:

- **Benchmark:** FRAMES, **fixed 20-sample subset** for fast, comparable runs:
  ```bash
  cd eval && uv run inspect eval wiki_eval/tasks.py@frames \
    --model anthropic/claude-haiku-4-5 --limit 20
  ```
  (`--model` here sets the **judge**; the agent model is set in `config.py`.)
- **Judge held constant** (Haiku) — so we measure the *agent*, not the grader.
- **One change per cycle** — isolate impact; carry forward only what helped.
- **Primary metric:** correctness (`model_graded_qa` accuracy). Also watch
  grounding **recall** and **wall-clock**.
- **Inspect failures one-by-one** with `eval/scripts/analyze_run.py` (latest run):
  prints per-sample correctness, recall, gold-vs-read pages, and step-cap count.
- **Stat caveat:** n=20 → stderr ≈ 0.10. Single-cycle deltas are **directional,
  not significant**. Confirm a winner on the full 100 (or 824) before locking it
  in.

**Verdict legend:** ✅ kept (improved) · ➖ neutral (no change, dropped or kept if
free) · ❌ dropped (regressed or not worth the cost).

---

## Rounds

### Round 1 — Harness & prompt (Haiku agent)

**Goal:** lift FRAMES correctness by tuning the loop and prompt. **Anchor:** 0.300.

| # | Change | Correctness | Recall | Verdict |
|---|--------|------------:|-------:|---------|
| 0 | Baseline (`max_steps=6`) | 0.300 | 0.375 | — |
| 1 | `max_steps` 6 → 20 | 0.300 | 0.455 | ✅ kept (enabler) |
| 2 | Decisive prompt rewrite | 0.450 | 0.444 | ✅ kept (**+0.15**) |
| 3 | Forced final answer on budget exhaustion | 0.450 | 0.492 | ✅ kept (neutral, removes wrong canned answers) |
| 4 | Fetch article **body** (drop `exintro`, 1500→4000) | **0.550** | 0.491 | ✅ kept (**+0.10**) |

**Story:** the baseline agent *ran out of steps and gave up* on most questions.
More steps fixed the give-up but not the answers — the real bottlenecks were
(a) the agent wandering instead of committing (fixed by the prompt) and (b) the
tool returning only the article intro, so even when it read the right page the
fact wasn't there (fixed by fetching the body). Net **0.30 → 0.55**.

### Round 2 — Model axis (anchor = R1 best, 0.550)

**Goal:** does a stronger model / more reasoning / output tweaks help?

| # | Isolated change | Correctness | Wall-clock | Verdict |
|---|-----------------|------------:|-----------:|---------|
| M1 | Haiku → **Sonnet 4.6**, no thinking | **0.750** | 1:34 | ✅ kept (**+0.20**) |
| M2 | + adaptive thinking (effort high) | 0.650 | 3:11 | ❌ dropped (−0.10, ~2× latency) |
| M2b | adaptive thinking, effort=low | 0.700 | 3:05 | ❌ dropped |
| M3 | + chain-of-thought in prompt | 0.700 | 1:43 | ❌ dropped (neutral) |
| M4 | + structured `ANSWER:` output format | 0.750 | 1:41 | ➖ reverted (neutral, clutters CLI) |

**Story:** the **model upgrade was the whole story** (+0.20, and faster than the
thinking variants). Everything meant to add *more reasoning* was redundant —
the multi-step tool loop already is the reasoning scaffold, so extended thinking
just doubled latency for no gain, and an explicit CoT prompt added nothing.
Kept only the Sonnet swap as a quality lever; the thinking/CoT/format changes
left no code behind (the temporary `THINKING`/`EFFORT` knobs were removed once
they showed no benefit).

---

## Open ideas / next levers

Not yet tried — candidates for the next round:

- **Confirm on the full set.** Re-run the R2 winner on 100 (and ideally 824) —
  current numbers are n=20.
- **Search quality.** Remaining failures include tangential-page retrieval; try
  `DEFAULT_SEARCH_LIMIT`↑ or letting the agent reformulate queries.
- **Numeric/temporal reasoning.** A cluster of failures is arithmetic over
  correctly-retrieved facts — a scratchpad/compute step or a stronger judge.
- **Stronger judge.** The judge is Haiku; verify it isn't the ceiling.

---

## How to add a round

1. Run the protocol above (fixed 20-sample subset, judge held constant, one
   change per cycle).
2. Append a new `### Round N — <axis> (anchor = <prev best>)` section using the
   template below.
3. Update **Biggest wins** and **Current best config** if your round changed the
   leader or the shipped settings.
4. Keep it short: a result table + a 2–4 sentence story. Detailed transcripts
   belong in the eval logs, not here.

Template:

```markdown
### Round N — <what you tuned> (anchor = <prev best correctness>)

**Goal:** <one line>

| # | Isolated change | Correctness | (Recall / Time) | Verdict |
|---|-----------------|------------:|----------------:|---------|
| 1 | ... | 0.xx | ... | ✅/➖/❌ |

**Story:** <2–4 sentences: what moved the needle, what didn't, what you kept.>
```
