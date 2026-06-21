# Parallel multi-query lookups — FRAMES impact report

**Date:** 2026-06-21
**Feature:** the `wikipedia` tool now accepts `queries`/`titles` lists that fan
out over the cached single-call path with bounded concurrency (≤3); the system
prompt nudges the agent to batch independent lookups.

## Setup

- Benchmark: **FRAMES**, fixed first **25** samples (multi-hop QA).
- Agent: **Haiku** (`claude-haiku-4-5`), pinned in-process for every run so a
  concurrent change to the shared `AGENT_MODEL` default couldn't skew the
  comparison. Judge: Haiku.
- Protocol: **4-run, warm-vs-warm** — cache cleared once, then baseline
  (warm-up + measured) and feature (warm-up + measured). Both arms measured on a
  warm cache so the cache isn't a confound.

## Results (baseline → feature)

| Metric | Baseline | Feature | Δ |
|--------|---------:|--------:|----:|
| Accuracy (correctness judge) | 0.680 | 0.600 | −0.080 |
| Retrieval recall | 0.510 | 0.484 | −0.026 |
| Avg steps (Sonnet/Haiku round-trips) | 12.48 | 12.20 | −0.28 |
| Avg tool calls / sample | 13.40 | 14.32 | +0.92 |
| Total tool calls | 335 | 358 | +23 |
| **Batched calls** | **0** | **1** | **+1** |
| **Batch-usage rate** | 0% | **0.3%** | — |
| Wall-clock (warm) | 90 s | 73 s | −17 s |

## Headline finding: the feature is correct but essentially **unused**

The agent issued **one** batched call across all 358 tool calls (0.3%). That one
call was a perfect fit — *"1990 census population of Oklahoma City, Tulsa, Broken
Arrow, Norman"* → fetched all four city articles in a single parallel call. The
mechanism works as designed. But with adoption at ~0%, **every headline delta
above is noise**, not signal:

- Accuracy −0.08 = 2 fewer correct out of 25, well inside the ±0.10 stderr.
- Steps essentially flat (12.48 → 12.20); tool calls slightly *up*.
- Wall-clock −17 s is run-to-run variance on a warm cache (cached fetches are
  ~instant, so in-tool parallelism saves ~nothing here), not a feature effect.

## Why adoption is near-zero

1. **FRAMES is sequentially dependent.** Most multi-hop questions chain: you must
   read article A to learn the entity you then search for in step B. Dependent
   hops *cannot* be parallelized, so the batchable pattern (several *independent*
   lookups known up front) is genuinely rare in this dataset — the Oklahoma-cities
   question is one of the few.
2. **Haiku doesn't proactively batch.** Even where a small independent fan-out was
   possible, the model defaulted to one-at-a-time despite the prompt nudge. A more
   capable model and/or a stronger prompt would likely raise adoption.

## Trace check

Inspected the feature run's tool inputs across all 25 samples: lists are
well-formed when used (the one batched call had 4 distinct, non-redundant
titles), and the singular path is unchanged. No malformed or 1-item lists, no
redundant repeats. The feature is being used *correctly* — just rarely.

## Verdict & recommendation

- **Gain on FRAMES: none measurable.** The capability is sound and free when
  idle, but FRAMES's sequential structure plus Haiku's behavior leave it almost
  unexercised, so it neither helps nor hurts here (within noise). Keep it — it is
  low-risk and pays off on workloads with independent fan-out (e.g. "compare these
  N entities", disambiguation across several candidate titles, the census-style
  aggregation question).
- **If we want to actually move FRAMES with this:** (a) re-run on **Sonnet**,
  which is likelier to plan independent fan-outs; (b) strengthen the prompt with a
  concrete example of when to batch; (c) consider auto-batching at the *loop*
  level — when the model emits multiple independent `tool_use` blocks in one turn,
  execute them concurrently (today the loop runs them sequentially). The real
  parallel-latency win shows up on a **cold** cache; with a warm cache the only
  lever is fewer round-trips, which requires the model to actually batch.

## Caveats

- 25-sample subset → noisy accuracy (±0.10 stderr).
- Haiku, not Sonnet (the shared default was Haiku at run time).
- Warm cache makes wall-clock insensitive to in-tool parallelism by design.
- Token figures are trajectory-approximate (the agent bypasses Inspect's model
  layer), so `avg_steps` is the reliable efficiency metric — and it barely moved.
