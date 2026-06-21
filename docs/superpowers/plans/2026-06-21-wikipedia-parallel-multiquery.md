# Parallel Multi-Query Wikipedia Lookups + FRAMES Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the agent submit several Wikipedia lookups in one tool call (`queries`/`titles` lists) executed in parallel inside the tool, then measure the FRAMES impact with a warm-vs-warm 4-run experiment on a 25-sample subset.

**Architecture:** The feature reuses the existing cached, backed-off single-call path (`search`/`get_article` → `_get`); two new fan-out functions run them concurrently via a thread pool (cap 3) over one shared httpx client. Dispatch routes list inputs to the fan-out functions. A standalone analyzer compares two Inspect `.eval` logs on cache-independent efficiency metrics.

**Tech Stack:** Python 3.12, httpx 0.28.1 (thread-safe client), `concurrent.futures.ThreadPoolExecutor`, Inspect AI (`inspect_ai.log.read_eval_log`), pytest. Agent-under-test = Sonnet 4.6; judge = Haiku.

## Global Constraints

- Feature code is `agent/` only; experiment tooling is `eval/` only. **Never** import from `eval/` in the agent. No change to `AgentResult` or `run()`.
- `eval` depends on `agent` via an **editable** install, so agent changes are live in the eval venv with no reinstall.
- Keep pure logic separate from I/O; unit-test all custom logic with fakes/monkeypatch. **No live network or API key in the test suite.**
- Config constants live in `agent/wiki_agent/config.py`. `MAX_CONCURRENCY = 3` (Wikimedia's documented concurrent ceiling), `MAX_BATCH = 10`.
- Tool functions return readable strings, never raise into the model.
- Git: stage only paths you changed; commit per task; messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Ordering matters:** Task 1 (baseline runs) must execute on the *current, pre-feature* code, before Tasks 2–5 change the agent.

---

### Task 1: Baseline FRAMES runs (pre-feature, warm-vs-warm)

**Files:** none modified. Produces two `.eval` logs; record the measured one's path.

**Interfaces:**
- Produces: the **baseline** `.eval` log path (used by Task 6's analysis).

- [ ] **Step 1: Clear the disk cache for a clean slate**

Run: `cd agent && uv run python -c "from wiki_agent import cache; print('cleared', cache.clear())"`
Expected: prints `cleared <N>`.

- [ ] **Step 2: Baseline warm-up run (populates cache, discarded)**

Run: `cd eval && uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5 --limit 25`
Expected: Inspect completes 25 samples; a new `.eval` appears in `eval/logs/`. This run is the warm-up (not measured).

- [ ] **Step 3: Baseline measured run (warm) and capture its path**

Run: `cd eval && uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5 --limit 25 && ls -t logs/*.eval | head -1`
Expected: completes 25 samples; the printed newest `.eval` path is the **baseline** log. Record it (e.g. copy to a note): `BASELINE_LOG=<path>`.

- [ ] **Step 4: No commit** (no files changed; logs are gitignored).

---

### Task 2: Config — concurrency + batch caps

**Files:**
- Modify: `agent/wiki_agent/config.py`

**Interfaces:**
- Produces: `config.MAX_CONCURRENCY:int=3`, `config.MAX_BATCH:int=10`.

- [ ] **Step 1: Add constants** after the existing cache constants in `config.py`

```python
# Parallel multi-query lookups: fan out a batched tool call over the cached
# single-call path. MAX_CONCURRENCY matches Wikimedia's documented concurrent
# ceiling; MAX_BATCH caps how many lookups one tool call may request.
MAX_CONCURRENCY = 3
MAX_BATCH = 10
```

- [ ] **Step 2: Verify import**

Run: `cd agent && uv run python -c "from wiki_agent import config; print(config.MAX_CONCURRENCY, config.MAX_BATCH)"`
Expected: `3 10`

- [ ] **Step 3: Commit**

```bash
git add agent/wiki_agent/config.py
git commit -m "Add MAX_CONCURRENCY/MAX_BATCH config for parallel lookups"
```

---

### Task 3: Parallel fan-out functions + schema + dispatch

**Files:**
- Modify: `agent/wiki_agent/wikipedia.py`
- Modify: `agent/tests/test_wikipedia.py`

**Interfaces:**
- Consumes: `config.MAX_CONCURRENCY`, `config.MAX_BATCH`, existing `search`/`get_article`.
- Produces: `wikipedia.search_many(queries:list[str], limit=…, *, client=None)->str`, `wikipedia.get_articles(titles:list[str], chars=…, *, client=None)->str`, `wikipedia._truncate(items)->tuple[list,str]`; updated `TOOL_SCHEMA` (`queries`/`titles` array fields) and `dispatch` routing.

- [ ] **Step 1: Write the failing tests** — append to `agent/tests/test_wikipedia.py`

```python
def test_truncate_caps_to_max_batch(monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH", 2)
    items, note = wikipedia._truncate(["a", "b", "c", "d"])
    assert items == ["a", "b"]
    assert "2 extra" in note


def test_truncate_no_note_within_limit():
    items, note = wikipedia._truncate(["a", "b"])
    assert items == ["a", "b"] and note == ""


def test_search_many_fans_out_in_order_with_shared_client(monkeypatch):
    calls = []

    def fake_search(q, limit, *, client=None):
        calls.append((q, limit, client))
        return f"R:{q}"

    monkeypatch.setattr(wikipedia, "search", fake_search)
    out = wikipedia.search_many(["a", "b", "c"], 5, client="shared")
    assert "=== search: 'a' ===\nR:a" in out
    assert out.index("R:a") < out.index("R:b") < out.index("R:c")
    assert [c[0] for c in calls] == ["a", "b", "c"]
    assert all(c[1] == 5 and c[2] == "shared" for c in calls)


def test_get_articles_fans_out(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_article", lambda t, chars, *, client=None: f"A:{t}:{chars}")
    out = wikipedia.get_articles(["X", "Y"], 100, client="shared")
    assert "=== article: 'X' ===\nA:X:100" in out
    assert "A:Y:100" in out


def test_search_many_truncates(monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH", 2)
    monkeypatch.setattr(wikipedia, "search", lambda q, limit, *, client=None: f"R:{q}")
    out = wikipedia.search_many(["a", "b", "c", "d"], client="s")
    assert "R:a" in out and "R:b" in out and "R:c" not in out
    assert "2 extra" in out


def test_dispatch_routes_search_many(monkeypatch):
    monkeypatch.setattr(wikipedia, "search_many", lambda qs, limit, *, client=None: f"many:{qs}:{limit}")
    out = wikipedia.dispatch({"action": "search", "queries": ["a", "b"], "limit": 3})
    assert out == "many:['a', 'b']:3"


def test_dispatch_routes_get_articles(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_articles", lambda ts, chars, *, client=None: f"arts:{ts}")
    out = wikipedia.dispatch({"action": "get_article", "titles": ["X", "Y"]})
    assert out == "arts:['X', 'Y']"


def test_dispatch_empty_queries_falls_back_to_single(monkeypatch):
    monkeypatch.setattr(wikipedia, "search", lambda q, limit, *, client=None: f"one:{q}")
    out = wikipedia.dispatch({"action": "search", "queries": [], "query": "z"})
    assert out == "one:z"


def test_schema_advertises_lists():
    props = wikipedia.TOOL_SCHEMA["input_schema"]["properties"]
    assert props["queries"]["type"] == "array"
    assert props["titles"]["type"] == "array"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd agent && uv run pytest tests/test_wikipedia.py -q`
Expected: FAIL — `_truncate`/`search_many`/`get_articles` undefined; schema lacks `queries`/`titles`.

- [ ] **Step 3: Add the array fields to `TOOL_SCHEMA`** in `wikipedia.py`

Update the tool description and add two properties. Change the `description` string to end with:

```python
        "you are unsure of the exact title. To look up several things at once, "
        "pass a `queries` list (with action='search') or a `titles` list (with "
        "action='get_article') in one call — they run in parallel."
```

And add inside `properties` (after `title`):

```python
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Several search terms to run in parallel in one call (use instead of 'query').",
            },
            "titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Several exact article titles to fetch in parallel in one call (use instead of 'title').",
            },
```

- [ ] **Step 4: Add the import and fan-out functions** in `wikipedia.py`

Add to the imports near the top:

```python
from concurrent.futures import ThreadPoolExecutor
```

Add in the "Public actions" section, after `get_article` and before `dispatch`:

```python
def _truncate(items: list[str]) -> tuple[list[str], str]:
    """Cap a batch to MAX_BATCH; return (kept, note) with a note when dropping."""
    if len(items) <= config.MAX_BATCH:
        return list(items), ""
    dropped = len(items) - config.MAX_BATCH
    return list(items[: config.MAX_BATCH]), (
        f"\n\n(Note: {dropped} extra item(s) beyond the {config.MAX_BATCH}-lookup "
        "limit were skipped.)"
    )


def search_many(queries: list[str], limit: int = config.DEFAULT_SEARCH_LIMIT, *,
                client: httpx.Client | None = None) -> str:
    """Run several searches in parallel; return labeled, concatenated results."""
    queries, note = _truncate(queries)
    owns = client is None
    client = client or _make_client()
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            results = list(pool.map(lambda q: search(q, limit, client=client), queries))
    finally:
        if owns:
            client.close()
    blocks = [f"=== search: {q!r} ===\n{r}" for q, r in zip(queries, results)]
    return "\n\n".join(blocks) + note


def get_articles(titles: list[str], chars: int = config.DEFAULT_EXTRACT_CHARS, *,
                 client: httpx.Client | None = None) -> str:
    """Fetch several articles in parallel; return labeled, concatenated results."""
    titles, note = _truncate(titles)
    owns = client is None
    client = client or _make_client()
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            results = list(pool.map(lambda t: get_article(t, chars, client=client), titles))
    finally:
        if owns:
            client.close()
    blocks = [f"=== article: {t!r} ===\n{r}" for t, r in zip(titles, results)]
    return "\n\n".join(blocks) + note
```

- [ ] **Step 5: Update `dispatch`** in `wikipedia.py` to route lists

```python
def dispatch(tool_input: dict, *, client: httpx.Client | None = None) -> str:
    """Route a tool-call payload to the right action. Returns a readable string."""
    action = tool_input.get("action")
    if action == "search":
        queries = tool_input.get("queries")
        if queries:
            return search_many(queries, tool_input.get("limit", config.DEFAULT_SEARCH_LIMIT), client=client)
        query = tool_input.get("query")
        if not query:
            return "Error: action='search' requires a 'query' or 'queries'."
        return search(query, tool_input.get("limit", config.DEFAULT_SEARCH_LIMIT), client=client)
    if action == "get_article":
        titles = tool_input.get("titles")
        if titles:
            return get_articles(titles, tool_input.get("chars", config.DEFAULT_EXTRACT_CHARS), client=client)
        title = tool_input.get("title")
        if not title:
            return "Error: action='get_article' requires a 'title' or 'titles'."
        return get_article(title, tool_input.get("chars", config.DEFAULT_EXTRACT_CHARS), client=client)
    return f"Error: unknown action '{action}'. Use 'search' or 'get_article'."
```

- [ ] **Step 6: Run to verify pass**

Run: `cd agent && uv run pytest tests/test_wikipedia.py -q`
Expected: PASS (all existing + 9 new). The existing `test_dispatch_requires_query_for_search` / `..._title_for_get_article` still pass (errors still start with "Error").

- [ ] **Step 7: Commit**

```bash
git add agent/wiki_agent/wikipedia.py agent/tests/test_wikipedia.py
git commit -m "Add parallel search_many/get_articles batch lookups to the tool"
```

---

### Task 4: Prompt nudge so the model adopts batching

**Files:**
- Modify: `agent/wiki_agent/agent.py`

**Interfaces:** none (prompt text only).

- [ ] **Step 1: Add one line to `SYSTEM_PROMPT`** in the "Work efficiently" block

Insert after the line about searching/reading the most relevant article:

```python
        "- When you need several independent lookups, request them together in "
        "one `wikipedia` call as a `queries` or `titles` list so they run in "
        "parallel — fewer round-trips than one at a time.\n"
```

- [ ] **Step 2: Verify the agent still imports and tests pass**

Run: `cd agent && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add agent/wiki_agent/agent.py
git commit -m "Nudge the agent to batch independent lookups into one call"
```

---

### Task 5: Log analyzer for the experiment

**Files:**
- Create: `eval/wiki_eval/analyze_runs.py`
- Create: `eval/tests/test_analyze_runs.py`

**Interfaces:**
- Produces: `analyze_runs._summarize_samples(records:list[dict])->dict`, `analyze_runs._record(sample)->dict`, `analyze_runs.summarize_log(path)->dict`, `analyze_runs.main(argv)`.

- [ ] **Step 1: Write the failing test** in `eval/tests/test_analyze_runs.py`

```python
"""Unit test for the experiment analyzer's pure aggregator — no logs/network."""

from wiki_eval.analyze_runs import _summarize_samples


def _rec(correct, steps, tool_inputs, tokens=0):
    return {"correct": correct, "steps": steps, "tool_inputs": tool_inputs, "tokens": tokens}


def test_summarize_counts_accuracy_steps_and_batch_usage():
    records = [
        _rec(True, 3, [{"action": "search", "queries": ["a", "b"]}, {"action": "get_article", "title": "X"}], tokens=100),
        _rec(False, 5, [{"action": "get_article", "titles": ["P", "Q", "R"]}], tokens=200),
    ]
    s = _summarize_samples(records)
    assert s["n"] == 2
    assert s["accuracy"] == 0.5
    assert s["avg_steps"] == 4.0
    assert s["total_tool_calls"] == 3
    assert s["batched_calls"] == 2          # the queries call + the titles call
    assert s["batch_usage_rate"] == 2 / 3
    assert s["avg_batch_size"] == 2.5       # (2 + 3) / 2
    assert s["approx_tokens"] == 300


def test_summarize_empty():
    s = _summarize_samples([])
    assert s["n"] == 0 and s["accuracy"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd eval && uv run pytest tests/test_analyze_runs.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eval/wiki_eval/analyze_runs.py`**

```python
"""Compare two FRAMES .eval logs (baseline vs feature) on efficiency metrics.

    uv run python -m wiki_eval.analyze_runs <baseline.eval> <feature.eval>

The agent calls Anthropic directly (bypassing Inspect's model layer), so token
usage is approximated from the trajectory and `avg_steps` (Sonnet round-trips)
is the reliable efficiency metric. The pure aggregator `_summarize_samples` is
unit-tested; log reading is thin I/O.
"""

from __future__ import annotations

import sys
from statistics import mean

CORRECT = "C"  # Inspect Score value for a correct model_graded_qa judgement


def _summarize_samples(records: list[dict]) -> dict:
    out = {"n": len(records), "accuracy": 0.0, "avg_steps": 0.0, "avg_tool_calls": 0.0,
           "total_tool_calls": 0, "batched_calls": 0, "batch_usage_rate": 0.0,
           "avg_batch_size": 0.0, "approx_tokens": 0}
    if not records:
        return out
    tool_inputs = [ti for r in records for ti in r["tool_inputs"]]
    batched = [ti for ti in tool_inputs if ti.get("queries") or ti.get("titles")]
    sizes = [len(ti.get("queries") or ti.get("titles")) for ti in batched]
    steps = [r["steps"] for r in records if r.get("steps") is not None]
    out.update(
        accuracy=sum(1 for r in records if r["correct"]) / len(records),
        avg_steps=mean(steps) if steps else 0.0,
        avg_tool_calls=mean([len(r["tool_inputs"]) for r in records]),
        total_tool_calls=len(tool_inputs),
        batched_calls=len(batched),
        batch_usage_rate=(len(batched) / len(tool_inputs)) if tool_inputs else 0.0,
        avg_batch_size=mean(sizes) if sizes else 0.0,
        approx_tokens=sum(r.get("tokens", 0) for r in records),
    )
    return out


def _record(sample) -> dict:
    """Flatten an Inspect EvalSample into the fields we aggregate on."""
    scores = getattr(sample, "scores", None) or {}
    mgqa = scores.get("model_graded_qa")
    correct = bool(mgqa is not None and getattr(mgqa, "value", None) == CORRECT)
    meta = getattr(sample, "metadata", None) or {}
    steps_list = (meta.get("trajectory") or {}).get("steps", [])
    tool_inputs = [s.get("tool_input") or {} for s in steps_list if s.get("kind") == "tool_call"]
    tokens = sum((s.get("input_tokens") or 0) + (s.get("output_tokens") or 0) for s in steps_list)
    return {"correct": correct, "steps": meta.get("steps"), "tool_inputs": tool_inputs, "tokens": tokens}


def _wall_clock(log) -> float | None:
    from datetime import datetime
    st = getattr(log.stats, "started_at", None)
    en = getattr(log.stats, "completed_at", None)
    if not st or not en:
        return None
    return (datetime.fromisoformat(en) - datetime.fromisoformat(st)).total_seconds()


def summarize_log(path: str) -> dict:
    from inspect_ai.log import read_eval_log
    log = read_eval_log(path)
    summary = _summarize_samples([_record(s) for s in (log.samples or [])])
    summary["wall_clock_s"] = _wall_clock(log)
    return summary


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: list[str]) -> None:
    baseline, feature = summarize_log(argv[1]), summarize_log(argv[2])
    keys = ["n", "accuracy", "avg_steps", "avg_tool_calls", "total_tool_calls",
            "batched_calls", "batch_usage_rate", "avg_batch_size", "approx_tokens",
            "wall_clock_s"]
    print(f"{'metric':<18} {'baseline':>12} {'feature':>12} {'delta':>12}")
    print("-" * 56)
    for k in keys:
        b, f = baseline.get(k), feature.get(k)
        delta = ""
        if isinstance(b, (int, float)) and isinstance(f, (int, float)):
            delta = _fmt(f - b)
        print(f"{k:<18} {_fmt(b):>12} {_fmt(f):>12} {delta:>12}")


if __name__ == "__main__":
    main(sys.argv)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd eval && uv run pytest tests/test_analyze_runs.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Full eval suite green**

Run: `cd eval && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add eval/wiki_eval/analyze_runs.py eval/tests/test_analyze_runs.py
git commit -m "Add FRAMES run analyzer (baseline vs feature efficiency metrics)"
```

---

### Task 6: Feature FRAMES runs + analysis + report

**Files:**
- Create: `docs/superpowers/reports/2026-06-21-parallel-multiquery-frames.md`

**Interfaces:**
- Consumes: the **baseline** log path from Task 1; `analyze_runs`.

- [ ] **Step 1: Feature warm-up run (caches new fetch patterns, discarded)**

Run: `cd eval && uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5 --limit 25`
Expected: completes 25 samples on the feature code (cache shared with baseline's per-title entries).

- [ ] **Step 2: Feature measured run (warm) and capture its path**

Run: `cd eval && uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5 --limit 25 && ls -t logs/*.eval | head -1`
Expected: completes; record the printed newest `.eval` as `FEATURE_LOG=<path>`.

- [ ] **Step 3: Run the analyzer on baseline vs feature**

Run: `cd eval && uv run python -m wiki_eval.analyze_runs "$BASELINE_LOG" "$FEATURE_LOG"`
Expected: a metric table with baseline, feature, delta columns. Capture it.

- [ ] **Step 4: Inspect 2–3 feature trajectories for batching quality**

Run: `cd eval && uv run python -c "
from inspect_ai.log import read_eval_log
log = read_eval_log('$FEATURE_LOG')
for s in (log.samples or [])[:3]:
    steps = (s.metadata.get('trajectory') or {}).get('steps', [])
    calls = [st.get('tool_input') for st in steps if st.get('kind')=='tool_call']
    print('Q:', s.input[:80]); [print('  ', c) for c in calls]; print()
"`
Expected: prints each sample's tool-call inputs; verify lists are well-formed and sensibly grouped (not 1-item lists, not redundant repeats).

- [ ] **Step 5: Write the concise report** to `docs/superpowers/reports/2026-06-21-parallel-multiquery-frames.md`

Include: the experiment setup (25-sample FRAMES, Sonnet agent, 4-run warm-vs-warm), the metric table from Step 3, the trace observations from Step 4, and a verdict — does the feature gain accuracy and/or efficiency (steps/tool-calls), does the agent use it well, and the warm-cache nuance (parallelism's wall-clock benefit mostly appears on a cold cache; with a warm cache the win is fewer turns). Keep it tight (≈1 page). Note the 25-sample noise caveat.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/reports/2026-06-21-parallel-multiquery-frames.md
git commit -m "Add FRAMES experiment report for parallel multi-query feature"
```

---

## Self-Review

**Spec coverage:**
- Schema `queries`/`titles` + description → Task 3. ✓
- `search_many`/`get_articles` (shared client, cap 3, order, truncation, headers) → Task 3. ✓
- Dispatch routing (lists + singular fallback) → Task 3. ✓
- Config `MAX_CONCURRENCY`/`MAX_BATCH` → Task 2. ✓
- Prompt nudge → Task 4. ✓
- 4-run warm-vs-warm protocol, 25 subset, clear cache → Tasks 1 (baseline) + 6 (feature). ✓
- Analyzer + pure `_summarize_samples` + metrics (accuracy, steps, tool calls, batch usage, approx tokens, wall-clock) → Task 5. ✓
- Trace inspection + concise report → Task 6. ✓
- Offline tests → Tasks 3, 5. ✓

**Placeholder scan:** No TBD/TODO; all code/commands concrete. The report content (Task 6 Step 5) is data-dependent by nature but its required sections are enumerated. ✓

**Type consistency:** `_summarize_samples` keys used in the test and `main` match the implementation; `_record` output keys (`correct`/`steps`/`tool_inputs`/`tokens`) match what `_summarize_samples` consumes; `search_many`/`get_articles`/`_truncate` signatures match Task 3 tests and dispatch calls. ✓

**Ordering note:** Task 1 runs the baseline on pre-feature code; Tasks 2–5 implement the feature; Task 6 runs the feature and compares. Correct.
