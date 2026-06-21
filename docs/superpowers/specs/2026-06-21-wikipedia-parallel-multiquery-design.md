# Wikipedia tool: parallel multi-query lookups + FRAMES impact experiment

**Date:** 2026-06-21
**Scope:** feature lives in `agent/`; experiment tooling in `eval/`. The one
allowed coupling (`eval → wiki_agent.run`) is unchanged; no change to
`AgentResult` or the `run()` signature.

## Problem & goal

FRAMES questions are multi-hop (2–15 article reads). Today the agent must issue
each Wikipedia lookup as its own tool call, so N lookups cost N agent turns
(N Sonnet round-trips). Give the agent an explicit way to **list several
queries/titles in one tool call**, executed **in parallel inside the tool**, and
measure the impact on FRAMES (accuracy + efficiency).

## Feature design (`agent/`)

Reuse the existing cached, backed-off single-call path; parallelism lives in the
tool (not the agent loop).

### Schema (`wikipedia.TOOL_SCHEMA`)

Add two optional array fields alongside the existing singulars:

- `queries: [string]` — multiple search terms, used with `action="search"`.
- `titles: [string]` — multiple exact titles, used with `action="get_article"`.

The tool description advertises: "To do several lookups at once, pass a
`queries` or `titles` list in a single call — they run in parallel."

### New functions (`wikipedia.py`)

```
search_many(queries: list[str], limit=DEFAULT_SEARCH_LIMIT, *, client=None) -> str
get_articles(titles: list[str], chars=DEFAULT_EXTRACT_CHARS, *, client=None) -> str
```

Both:
- Truncate the input list to `config.MAX_BATCH` (10); if truncated, append a
  one-line note naming how many were dropped.
- Create **one shared `httpx.Client`** (httpx clients are thread-safe) and fan
  out over the existing `search`/`get_article` with
  `ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY)`, passing the shared
  client so each underlying call reuses the pool and does **not** close it.
- Preserve input order (`list(pool.map(...))`).
- Concatenate the per-item readable strings under clear headers, e.g.
  `=== search: 'Harriet Lane' ===\n<result>` separated by blank lines.

Because each item goes through the unchanged `search`/`get_article` → `_get`, it
independently uses the disk cache + Retry-After/exponential backoff, and **shares
cache keys with single-item calls** (so a warm cache from the baseline benefits
the feature run — clean experiment).

### Dispatch (`wikipedia.dispatch`)

- `action="search"`: if `queries` is a non-empty list → `search_many`; elif
  `query` → `search`; else the existing error.
- `action="get_article"`: if `titles` is a non-empty list → `get_articles`; elif
  `title` → `get_article`; else the existing error.

Singular routing and its tests are untouched.

### Config (`config.py`)

```python
MAX_CONCURRENCY = 3   # Wikimedia's documented concurrent-request ceiling
MAX_BATCH = 10        # max lookups fanned out from one batched tool call
```

### Prompt nudge (`agent.py` SYSTEM_PROMPT)

Add one concise line so the model actually adopts the feature (essential for a
fair measurement):

> "When you need several independent lookups, pass them together in one
> `wikipedia` call as a `queries` or `titles` list so they run in parallel —
> fewer round-trips than one-at-a-time."

No agent-loop change: a single batched tool call now performs N parallel
lookups; the loop still executes tool-use blocks as before.

## Experiment protocol (`eval/`) — 4-run, warm-vs-warm, 25-sample subset

Agent under test = its own default **Sonnet 4.6**; judge = Haiku. Fixed first-25
FRAMES slice via `--limit 25` (deterministic, identical across runs).

1. **Clear cache:** `uv run --project ../agent python -c "from wiki_agent import cache; print(cache.clear())"` (or `wiki-agent ask --clear-cache`), and remove stale logs we might confuse.
2. **Baseline warm-up** (current code), `--limit 25` → populates cache *(log discarded)*.
3. **Baseline measured** (warm) → keep this `.eval` log as **baseline**.
4. **Implement the feature.**
5. **Feature warm-up**, `--limit 25` → caches any new fetch patterns *(discarded)*.
6. **Feature measured** (warm) → keep this `.eval` log as **feature**.

Run command (steps 2–3, 5–6):
```
cd eval && uv run inspect eval wiki_eval/tasks.py@frames \
  --model anthropic/claude-haiku-4-5 --limit 25
```
Warm-up vs measured runs are distinguished by capturing the log filename printed
by Inspect (newest `.eval` in `eval/logs/` after each run).

Rationale for warming **both** arms: otherwise baseline is cold and feature is
warm, so any wall-clock delta is mostly the cache, not the feature. Cache-
independent metrics (below) would be valid either way, but warm-vs-warm makes
wall-clock honest too.

Caveat recorded in the report: Sonnet is non-deterministic, so runs explore
slightly different pages → the warm cache is "mostly", not "perfectly", warm;
and a 25-sample subset makes accuracy deltas noisy.

## Analysis & report

`eval/analyze_runs.py` (standalone, run as `uv run python analyze_runs.py
<baseline.eval> <feature.eval>`):

- Reads both logs with `inspect_ai.log.read_eval_log`.
- Pure `_summarize_samples(samples) -> dict` (unit-tested with fakes) computes:
  - **accuracy** — mean of the `correctness` scorer (C=1/I=0),
  - **avg steps** — mean of `sample.metadata["steps"]`,
  - **avg tool calls** — mean count of `TOOL_CALL` entries in
    `sample.metadata["trajectory"]["steps"]`,
  - **batch usage** — fraction of tool calls whose input contains a non-empty
    `queries`/`titles` list, and the mean list length when batched,
  - **agent tokens** — summed input/output usage for the Sonnet model.
- A separate I/O helper pulls wall-clock from the log stats.
- Prints a side-by-side table with deltas (baseline → feature).

**Trace inspection:** print 2–3 feature trajectories' tool-call inputs to verify
the lists are well-formed and sensibly grouped (not 1-item lists, not redundant
repeats), and include the observation in the report.

**Deliverable:** a very concise report (committed under `docs/`) stating whether
the feature gains accuracy and/or efficiency and whether the agent uses it well,
with the headline numbers and the warm-cache nuance (parallelism's wall-clock
benefit shows mostly on a cold cache; with a warm cache the win is fewer
turns/tokens).

## Metrics summary

| Metric | Source | Cache-sensitive? | Role |
|--------|--------|------------------|------|
| FRAMES accuracy | correctness scorer | no | primary |
| avg steps | `metadata["steps"]` | no | primary |
| avg tool calls | trajectory | no | primary |
| agent tokens | log model_usage (Sonnet) | no | primary |
| batch-usage rate / size | trajectory tool inputs | no | adoption |
| wall-clock | log stats | yes (warm-vs-warm) | secondary |

## Tests (offline, no network/API key)

- `search_many`: monkeypatch `wikipedia.search` to record calls; assert all
  queries run, order preserved, headers + concatenation correct, one shared
  client passed (owns=False so singles don't close it), truncation note past
  `MAX_BATCH`.
- `get_articles`: same shape against `wikipedia.get_article`.
- `dispatch`: `{action:"search", queries:[...]}` → `search_many`;
  `{action:"get_article", titles:[...]}` → `get_articles`; empty list falls back
  to singular/error; singular routing still works.
- `_summarize_samples`: fake sample dicts → expected accuracy/steps/tool-call/
  batch-usage aggregation.

## Non-goals

- No single-request `titles=A|B|C` batching (rejected: coarse cache keys not
  shared with per-title singles; complicates the warm-cache experiment).
- No agent-loop concurrency for separate tool-use blocks (the feature is the
  list interface; loop unchanged).
- No change to other benchmarks, scorers, or the agent↔eval boundary.

## Files touched

| Path | Change |
|------|--------|
| `agent/wiki_agent/config.py` | `MAX_CONCURRENCY`, `MAX_BATCH` |
| `agent/wiki_agent/wikipedia.py` | schema fields; `search_many`/`get_articles`; dispatch routing |
| `agent/wiki_agent/agent.py` | one-line SYSTEM_PROMPT nudge |
| `agent/tests/test_wikipedia.py` | batch + dispatch tests |
| `eval/analyze_runs.py` | **New** — log comparison + pure summarizer |
| `eval/tests/test_analyze_runs.py` | **New** — `_summarize_samples` test |
| `docs/superpowers/specs/2026-06-21-wikipedia-parallel-multiquery-design.md` | this spec |
| `docs/.../<report>.md` | the concise experiment report (deliverable) |
