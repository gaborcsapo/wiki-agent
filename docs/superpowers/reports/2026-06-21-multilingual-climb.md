# Multilingual climb — `multilingual_qa` correctness report

**Date:** 2026-06-21
**Goal:** raise correctness on `multilingual_qa` (30 samples) by improving the
Wikipedia tool over 3 iterative improve→measure cycles.
**Setup:** agent = **Haiku** (`claude-haiku-4-5`, pinned in-process); judge =
Haiku; one run per cycle (correctness is cache-independent). Per-cycle failures
were inspected to choose the next change.

## Results

| Stage | Change | Accuracy | cross_lingual_fact | richer_native_page | foreign_language_query |
|-------|--------|---------:|-----:|-----:|-----:|
| Baseline | English-only tool | **0.133** | 0.08 | 0.30 | 0.00 |
| **Cycle 1** | **`lang` param → query any-language edition** | **0.333** | 0.08 | 0.40 | **0.62** |
| Cycle 2 | search returns intro extracts | 0.267 | 0.00 | 0.30 | 0.62 |
| Cycle 3 | native-first prompt | 0.267 (×2) | 0.00–0.08 | 0.30–0.40 | 0.50 |
| **Final (= Cycle 1)** | C2 & C3 reverted | **0.333** | 0.08 | 0.40 | 0.62 |

Noise: 30 samples → ~±0.09 stderr overall; the small per-category counts (8–12)
make single-cycle deltas of 1–2 samples statistically insignificant.

## Which change made the biggest improvement

**Cycle 1 — multilingual `lang` support — delivered the entire gain: 0.133 →
0.333 (+0.20, ~2.5×).** Letting the agent query the native-language Wikipedia
edition (`lang=hu/is/et/...`, with the API host and cache key keyed by language)
fixed the dominant failure: the English-only tool simply couldn't reach facts
that live only on native editions. The largest lift was on
`foreign_language_query` (0% → 62%) — when the question is in Hungarian/etc., the
agent now searches that edition.

This is the only statistically robust result. **Cycles 2 and 3 did not improve on
Cycle 1 and were reverted:**

- **Cycle 2 (search returns intro extracts):** regressed to 0.267. Trace
  inspection showed the lead snippet *diverted* the agent — e.g. for Grímur
  Thomsen it chased a tangential "Bessastaðir" page and never opened the full
  Icelandic article that held the fact. Reverted.
- **Cycle 3 (native-first prompt):** 0.267 on two runs — no gain over Cycle 1,
  slightly worse. Pushing "native-first, skip English" didn't move correctness
  (and removed a useful English fallback). Reverted; YAGNI.

## Where we are at the end

**Final: 0.333 (the Cycle-1 tool), up from 0.133 baseline — a 2.5× improvement.**
The shipped change is the `lang` parameter end-to-end (schema, search/get_article
+ batch variants, dispatch, per-language API host, language-aware cache key) plus
a prompt line telling the agent to use the native edition.

**Remaining failure mode — `cross_lingual_fact` (~8%).** Diagnosis showed the
agent usually *reaches* the right native article and the fact is *present* in the
returned extract, yet it still misses — it exhausts the 6-step budget on
detours, or doesn't extract a specific fact from Hungarian/Icelandic/Estonian
prose. This is bounded by **Haiku's foreign-language comprehension and the step
budget**, not by the tool's reach, so further *tool* changes have limited
leverage here.

**Highest-EV next steps (beyond this 3-cycle tool experiment):** run the agent on
**Sonnet** (stronger multilingual reading), and/or raise `multilingual_qa`'s step
budget (an eval/solver knob, deliberately left fixed here to keep the comparison
fair). Both target the actual ceiling — comprehension and budget — rather than
tool reach, which Cycle 1 already solved.

## Caveats

- 30-sample subset, single model (Haiku) → noisy; only the +0.20 Cycle-1 gain is
  significant.
- A concurrent change set the shared `AGENT_MODEL` default to Haiku; all runs
  were pinned to Haiku in-process so the comparison is consistent.
