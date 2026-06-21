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
| — | **Accuracy levers** (search 5→10, extract→6000, steps→30, tokens→4096) | **+0.10 Haiku / −0.05 Sonnet** | R5 |

**Model-dependent lesson (R5):** "turn the knobs up" levers help the **weak**
model (Haiku) but **not the strong one** (Sonnet) — Sonnet already finds the
right page from fewer results and a smaller extract, so extra context is
distraction, not signal. Tune levers to the model tier.

**Things that did _not_ help** (tested and dropped): extended/adaptive thinking,
chain-of-thought prompting, structured `ANSWER:` output format (R2); parallel
multi-query lookups (R3, no FRAMES lift); search-returns-extracts and a
native-first prompt (R4, reverted).

**On other benchmarks:** the biggest win on **`multilingual_qa`** was
multilingual `lang` support — querying the native-language Wikipedia edition —
which drove **+0.20** (0.133 → 0.333, ~2.5×). See **Round 4**.

---

## Current best config

The settings the agent ships with today (all in `agent/wiki_agent/`).

| Setting | Value | Where | Why |
|--------|-------|-------|-----|
| Agent model | `claude-haiku-4-5` (default) | `config.py` `AGENT_MODEL` | cheap/fast for dev |
| — quality lever | `claude-sonnet-4-6` on demand | `run(model=...)` | +0.20 correctness (R2) |
| Step budget | `max_steps=30` | `eval/.../tasks.py` `frames()` | FRAMES needs many reads; 6→20→30 (R1, R5) |
| Article fetch | body, `exchars=6000` | `wikipedia.get_article` | facts below the lead; 1500→4000→6000 (R1, R5) |
| Search breadth | `DEFAULT_SEARCH_LIMIT=10` | `config.py` | more candidate titles; helps Haiku (R5) |
| Response tokens | `MAX_TOKENS=4096` | `config.py` | headroom for long answers (R5) |
| Budget-exhaustion | forced final answer | `agent.run` | no canned non-answers (R1) |
| System prompt | decisive v2 | `agent.SYSTEM_PROMPT` | stop looping, commit (R1) |
| Multilingual | `lang` per-edition querying | `wikipedia` + prompt | +0.20 on multilingual_qa (R4) |
| Parallel lookups | `queries`/`titles` lists, conc≤3 | `wikipedia` | kept (free when idle); no FRAMES lift (R3) |
| Networking | maxlag + Retry-After/exp backoff | `wikipedia._get` | resilience under rate limits (infra) |
| Cache | isolated disk cache, lang-keyed | `wikipedia.cache` | cheap benchmark re-runs (infra) |

**Score on the 20-sample FRAMES subset:** Haiku (default, with R5 levers) ≈
**0.65**; Sonnet ≈ **0.80**. Grounding recall ≈ 0.50. The R5 levers help Haiku
(+0.10) but not Sonnet (−0.05) — see R5. (Extended thinking was tested and is not
used — no gain, ~2× latency; see R2.)

**Score on the 30-sample `multilingual_qa` subset (Haiku):** **0.13 → 0.33**
after adding `lang` support (R4).

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

### Round 3 — Parallel multi-query lookups (FRAMES, 25-sample, Haiku)

**Goal:** cut agent round-trips by letting the model batch several lookups into
one tool call. **Protocol:** 4-run **warm-vs-warm** (cache cleared once, then
baseline warm-up+measured, feature warm-up+measured) so the disk cache isn't a
confound; agent pinned to Haiku. **Anchor:** baseline 0.680.

| # | Isolated change | Correctness | Batch adoption | Verdict |
|---|-----------------|------------:|---------------:|---------|
| 1 | `queries`/`titles` list inputs, parallel fan-out (conc≤3) | 0.600 | **1 / 358 tool calls** | ➖ kept, no lift |

**Story:** the feature works — the one time the model used it, it correctly
fetched four Oklahoma-city articles in a single call — but it went essentially
**unused** (0.3% of tool calls), so the −0.08 is pure noise. Two reasons:
FRAMES multi-hop questions are **sequentially dependent** (each hop needs the
previous answer, so there's nothing to parallelize), and Haiku doesn't
proactively batch even when it could. With a warm cache the only possible win is
fewer round-trips, which never materialized. Kept anyway: it's low-risk, free
when idle, and pays off on independent-fan-out workloads — just not FRAMES.

### Round 4 — Multilingual support (`multilingual_qa`, 30-sample, Haiku)

**Goal:** close the multilingual gap — the English-only tool can't reach facts
that live on native-language Wikipedias. **Protocol:** 3 iterative cycles, one
tool change each, one run per cycle (correctness is cache-independent), failures
inspected between cycles. **Anchor:** baseline 0.133.

| # | Isolated change | Correctness | Verdict |
|---|-----------------|------------:|---------|
| C1 | `lang` param → query any-language edition (host + cache key per-lang) | **0.333** | ✅ kept (**+0.20**) |
| C2 | search returns intro extracts (`generator=search`) | 0.267 | ❌ reverted (diverted the agent) |
| C3 | native-first prompt ("skip English, read full article") | 0.267 (×2) | ❌ reverted (no gain) |

**Story:** **C1 was the whole story** — letting the agent set `lang` to the
native edition took correctness from 0.13 → 0.33 (~2.5×), with the biggest lift
on questions *asked* in another language (`foreign_language_query` 0% → 62%). C2
backfired: the lead snippet *diverted* the agent away from opening the full
native article that held the fact (e.g. it chased a tangential page for Grímur
Thomsen and never read his Icelandic bio), so it was reverted. C3's native-first
prompt was neutral-to-worse over two runs and dropped (YAGNI). The remaining
`cross_lingual_fact` failures (~8%) are **not** tool-reach problems — the agent
reaches the native article and the fact is in the returned extract — they're
bounded by **Haiku's foreign-language comprehension + the 6-step budget**. Next
levers there: Sonnet, or a larger step budget. Full write-up:
`docs/superpowers/reports/2026-06-21-multilingual-climb.md`.

### Round 5 — Accuracy levers, Haiku vs Sonnet (FRAMES, 20-sample)

**Goal:** do "turn the knobs up" levers improve correctness, and does it depend
on the model? **Changes (all bumped together):** `DEFAULT_SEARCH_LIMIT` 5→10,
`DEFAULT_EXTRACT_CHARS` 4000→6000, FRAMES `max_steps` 20→30, `MAX_TOKENS`
2048→4096. Ran the same config on both tiers.

| | Haiku | Sonnet |
|------------|------:|-------:|
| baseline | 0.55 | 0.80 |
| + levers | **0.65** | 0.75 |
| Δ | **+0.10** ✅ | −0.05 ➖ |

**Story:** the levers **help the weak model and not the strong one.** Haiku gains
+0.10 — more candidate titles and a longer extract give it more chances to land
on the right page and pull the fact. Sonnet *regresses* slightly (within noise):
it already nails the right page from 5 results and a 4k extract, so 10 results +
6k chars is distraction, not signal. **Kept as the default** because the shipped
default is Haiku; cost is ~2× wall-clock (bigger extracts bust the read cache).
On Sonnet, prefer the leaner settings — the levers don't earn their cost there.
Also notable: the current Sonnet baseline (0.80) beats R2's (0.75) — the R3/R4
parallel + multilingual work lifted Sonnet too. Caveat: this was a **combined**
bump, not isolated per-lever (the user asked to raise all at once); if precise
per-lever attribution matters, isolate in a follow-up.

### Infrastructure that made the climbing cheap (no correctness delta)

Not a correctness lever, but part of the experience: the tool gained an
**isolated disk cache** (raw API JSON, language-keyed) plus **maxlag +
Retry-After/exponential backoff**. Together they let benchmarks be re-run many
times without re-fetching or tripping Wikipedia's rate limiter — essential when
hill-climbing means dozens of runs. A separate `ratelimit_bench.py` empirically
picked the best anonymous client config (serial + compliant User-Agent →
~3.3 req/s, <1% throttling). A **model-pinning runner** (`eval/run_pinned.py`)
was added after a concurrent edit flipped the shared `AGENT_MODEL` mid-experiment
— it pins the agent model in-process so a run can't be skewed by on-disk changes.
Per-run analyzers (`analyze_runs.py`, `analyze_multilingual.py`) print accuracy,
per-category/-language breakdowns, and failure lists to drive the next cycle.

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
- **Multilingual ceiling (R4).** `cross_lingual_fact` is comprehension/budget
  bound, not tool-reach bound: re-run multilingual on **Sonnet** and/or raise
  `multilingual_qa`'s step budget (it's still 6; FRAMES uses 20). Also worth
  trying: `langlinks` to pivot en→native titles directly.
- **Parallel lookups on a non-sequential benchmark (R3).** The batch feature was
  idle on FRAMES because the hops are dependent; it may help on tasks with
  independent fan-out (compare-N-entities, disambiguation) — measure there.

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
