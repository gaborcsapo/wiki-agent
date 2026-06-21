# FRAMES Retrieval-Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the FRAMES benchmark to the eval suite with a FRAMES-only retrieval-grounding scorer that reports page-overlap recall (headline) plus precision/F1 (diagnostics).

**Architecture:** A new `retrieval_grounding` Inspect scorer compares the set of Wikipedia pages the agent actually read (parsed from `tool_result` steps in the trajectory) against FRAMES's gold `reference_pages` (carried in sample metadata by the FRAMES dataset loader). Grounding stays FRAMES-only purely by composition — only the `frames()` task lists the scorer, and only the FRAMES loader populates the metadata it reads. The agent is not modified.

**Tech Stack:** Python 3.12+, Inspect AI (`inspect_ai`), pytest, `datasets` (dev-only, for the one-time dataset build).

## Global Constraints

- Python `>=3.12`.
- **No live API or network calls in tests.** All tests must pass with no `ANTHROPIC_API_KEY` set.
- Separate pure logic from I/O — parsing/scoring helpers are pure and unit-tested with fakes.
- Model ids live in `eval/wiki_eval/config.py`; never hardcode them elsewhere.
- The agent must not import from `eval/`. This plan touches `eval/` only; the agent is untouched.
- Git: stage only the exact paths you changed (no `git add -A` / `git add .`). Review `git diff --cached` before committing.
- End every commit message with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- All commands run from `eval/` (its own `uv` venv) unless noted.

---

### Task 1: Pure grounding helpers

**Files:**
- Modify: `eval/wiki_eval/scorers.py` (add imports + three pure helpers)
- Test: `eval/tests/test_grounding.py` (create)

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `_normalize_wiki_url(url: str) -> str | None` — canonical lowercased slug (spaces, not underscores) for a `/wiki/<slug>` URL; `None` for non-article URLs.
  - `_fetched_pages(steps: list[dict]) -> set[str]` — normalized slugs read via `get_article`, extracted from `tool_result` step contents.
  - `_grounding_scores(gold: set[str], read: set[str]) -> dict[str, float]` — `{"recall", "precision", "f1"}`.

- [ ] **Step 1: Write the failing tests**

Create `eval/tests/test_grounding.py`:

```python
"""Unit tests for the pure retrieval-grounding helpers (no API/network)."""

from wiki_eval.scorers import (
    _fetched_pages,
    _grounding_scores,
    _normalize_wiki_url,
)


def test_normalize_basic_and_underscores():
    assert _normalize_wiki_url("https://en.wikipedia.org/wiki/James_Buchanan") == "james buchanan"


def test_normalize_percent_encoding():
    assert _normalize_wiki_url("https://en.wikipedia.org/wiki/Beyonc%C3%A9") == "beyoncé"


def test_normalize_strips_query_and_fragment():
    url = "https://en.wikipedia.org/wiki/Apollo_11?foo=bar#Crew"
    assert _normalize_wiki_url(url) == "apollo 11"


def test_normalize_non_article_returns_none():
    assert _normalize_wiki_url("https://en.wikipedia.org/w/index.php?title=X") is None
    assert _normalize_wiki_url("not a url") is None


def _result(content):
    return {"kind": "tool_result", "content": content}


def test_fetched_pages_collects_get_article_urls():
    steps = [
        {"kind": "tool_call", "tool_input": {"action": "get_article", "title": "Apollo 11"}},
        _result("Apollo 11\nhttps://en.wikipedia.org/wiki/Apollo_11\n\nApollo 11 was..."),
    ]
    assert _fetched_pages(steps) == {"apollo 11"}


def test_fetched_pages_ignores_search_and_errors():
    steps = [
        _result("1. Apollo 11 — first crewed Moon landing\n2. Apollo program — ..."),
        _result("No Wikipedia article titled 'Xyz' exists. Try action='search'."),
    ]
    assert _fetched_pages(steps) == set()


def test_fetched_pages_dedupes():
    line = "X\nhttps://en.wikipedia.org/wiki/X\n\nbody"
    assert _fetched_pages([_result(line), _result(line)]) == {"x"}


def test_grounding_perfect():
    s = _grounding_scores({"a", "b"}, {"a", "b"})
    assert s == {"recall": 1.0, "precision": 1.0, "f1": 1.0}


def test_grounding_partial_recall():
    s = _grounding_scores({"a", "b"}, {"a"})
    assert s["recall"] == 0.5
    assert s["precision"] == 1.0


def test_grounding_over_exploration_lowers_precision_not_recall():
    s = _grounding_scores({"a", "b"}, {"a", "b", "c", "d"})
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5


def test_grounding_zero_and_empty_gold():
    assert _grounding_scores({"a"}, set()) == {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    assert _grounding_scores(set(), {"a"}) == {"recall": 0.0, "precision": 0.0, "f1": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalize_wiki_url' from 'wiki_eval.scorers'`.

- [ ] **Step 3: Write the helpers**

Add to the top of `eval/wiki_eval/scorers.py`, after the existing `from __future__ import annotations` line:

```python
import re
from urllib.parse import unquote, urlsplit
```

Add these helpers to `eval/wiki_eval/scorers.py` (above `correctness_judge`):

```python
_WIKI_PATH_RE = re.compile(r"^/wiki/(.+)$")
_WIKI_URL_RE = re.compile(r"https?://en\.wikipedia\.org/wiki/\S+")


def _normalize_wiki_url(url: str) -> str | None:
    """Canonical slug for an English-Wikipedia article URL, else None.

    Lowercased, spaces not underscores, percent-decoded, query/fragment dropped.
    """
    parts = urlsplit(url.strip())
    match = _WIKI_PATH_RE.match(parts.path)
    if not match:
        return None
    slug = unquote(match.group(1)).replace("_", " ").strip().casefold()
    return slug or None


def _fetched_pages(steps: list[dict]) -> set[str]:
    """Normalized slugs the agent actually read via get_article.

    Only get_article results carry a canonical `.../wiki/<slug>` URL line; search
    listings and error messages do not, so they contribute nothing.
    """
    pages: set[str] = set()
    for step in steps:
        if step.get("kind") != "tool_result":
            continue
        for raw in _WIKI_URL_RE.findall(step.get("content") or ""):
            slug = _normalize_wiki_url(raw.rstrip(".,);"))
            if slug:
                pages.add(slug)
    return pages


def _grounding_scores(gold: set[str], read: set[str]) -> dict[str, float]:
    """Recall (headline), precision, and F1 of read pages vs. gold pages."""
    if not gold:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(gold & read)
    recall = hits / len(gold)
    precision = hits / len(read) if read else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: PASS (all 10 tests).

- [ ] **Step 5: Commit**

```bash
git add wiki_eval/scorers.py tests/test_grounding.py
git commit -m "Add pure retrieval-grounding helpers (normalize, fetched-pages, scores)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `retrieval_grounding` scorer

**Files:**
- Modify: `eval/wiki_eval/scorers.py` (add the `@scorer`)
- Test: `eval/tests/test_grounding.py` (append scorer test)

**Interfaces:**
- Consumes: `_normalize_wiki_url`, `_fetched_pages`, `_grounding_scores` (Task 1); reads `state.metadata["reference_pages"]` (list of URLs) and `state.metadata["trajectory"]["steps"]`.
- Produces: `retrieval_grounding()` — an Inspect scorer returning `Score(value={"recall","precision","f1"})`.

- [ ] **Step 1: Write the failing test**

Append to `eval/tests/test_grounding.py`:

```python
import asyncio
from types import SimpleNamespace

from wiki_eval.scorers import retrieval_grounding


def test_retrieval_grounding_scorer_end_to_end():
    steps = [
        {"kind": "tool_result", "content": "Apollo 11\nhttps://en.wikipedia.org/wiki/Apollo_11\n\nbody"},
        {"kind": "tool_result", "content": "Neil Armstrong\nhttps://en.wikipedia.org/wiki/Neil_Armstrong\n\nbody"},
    ]
    state = SimpleNamespace(metadata={
        "reference_pages": [
            "https://en.wikipedia.org/wiki/Apollo_11",
            "https://en.wikipedia.org/wiki/Buzz_Aldrin",
        ],
        "trajectory": {"steps": steps},
    })
    score = asyncio.run(retrieval_grounding()(state, None))
    # Read Apollo 11 (gold) + Neil Armstrong (not gold); missed Buzz Aldrin.
    assert score.value["recall"] == 0.5
    assert score.value["precision"] == 0.5
    assert score.metadata["n_hit"] == 1


def test_retrieval_grounding_handles_missing_metadata():
    state = SimpleNamespace(metadata={})
    score = asyncio.run(retrieval_grounding()(state, None))
    assert score.value["recall"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding.py::test_retrieval_grounding_scorer_end_to_end -v`
Expected: FAIL — `ImportError: cannot import name 'retrieval_grounding'`.

- [ ] **Step 3: Write the scorer**

Extend the existing scorer imports in `eval/wiki_eval/scorers.py`. The current import block is:

```python
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
```

Replace it with (adds `mean`):

```python
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    mean,
    model_graded_qa,
    scorer,
    stderr,
)
```

Add the scorer at the end of `eval/wiki_eval/scorers.py`:

```python
@scorer(metrics={"recall": [mean(), stderr()], "precision": [mean()], "f1": [mean()]})
def retrieval_grounding():
    """FRAMES-only: overlap between pages the agent read and gold reference pages.

    Recall is the headline (did the agent find the needed evidence?); precision
    and F1 ride along as diagnostics. Gold pages come from the sample metadata
    that only the FRAMES loader populates, so this scorer is inert elsewhere.
    """

    async def score(state: TaskState, target: Target) -> Score:
        gold = {
            slug
            for url in state.metadata.get("reference_pages", [])
            if (slug := _normalize_wiki_url(url))
        }
        steps = state.metadata.get("trajectory", {}).get("steps", [])
        read = _fetched_pages(steps)
        values = _grounding_scores(gold, read)
        hits = len(gold & read)
        return Score(
            value=values,
            explanation=f"{hits}/{len(gold)} gold pages read; {len(read)} read total",
            metadata={"n_gold": len(gold), "n_read": len(read), "n_hit": hits},
        )

    return score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: PASS (all tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add wiki_eval/scorers.py tests/test_grounding.py
git commit -m "Add retrieval_grounding scorer (recall headline + precision/F1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: FRAMES dataset converter + committed subsample

**Files:**
- Create: `eval/scripts/build_frames.py` (dev-only converter)
- Create: `eval/wiki_eval/datasets/frames.jsonl` (generated, committed)
- Modify: `eval/pyproject.toml` (add `datasets` to the dev dependency group)

**Interfaces:**
- Consumes: nothing in-repo.
- Produces: `frames.jsonl` rows shaped `{"input": str, "target": str, "reference_pages": list[str]}` — consumed by Task 4.

> Note: this script makes a network call to Hugging Face. It is **not** imported by the package and **not** run by tests; it is a one-time/refresh dev tool. Tests stay offline because Task 4 reads the committed jsonl (and unit-tests the loader purely).

- [ ] **Step 1: Add the dev dependency**

In `eval/pyproject.toml`, change the dev group from:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

to:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "datasets>=2.0",
]
```

Then run: `uv sync`
Expected: resolves and installs `datasets`.

- [ ] **Step 2: Write the converter script**

Create `eval/scripts/build_frames.py`:

```python
"""Dev-only: build wiki_eval/datasets/frames.jsonl from google/frames-benchmark.

Not imported by the package and not covered by tests. Requires the `datasets`
dev dependency and network access.

Usage (from eval/):
    uv run python scripts/build_frames.py --limit 100
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "wiki_eval" / "datasets" / "frames.jsonl"


def _reference_pages(wiki_links) -> list[str]:
    """FRAMES stores reference URLs as a stringified Python list; parse to a list."""
    if isinstance(wiki_links, str):
        return list(ast.literal_eval(wiki_links))
    return list(wiki_links)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Rows to keep (0 = all 824).")
    args = parser.parse_args()

    dataset = load_dataset("google/frames-benchmark", split="test")
    rows = []
    for i, record in enumerate(dataset):
        if args.limit and i >= args.limit:
            break
        rows.append(
            {
                "input": record["Prompt"],
                "target": record["Answer"],
                "reference_pages": _reference_pages(record["wiki_links"]),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the committed subsample**

Run: `uv run python scripts/build_frames.py --limit 100`
Expected: `Wrote 100 rows to .../eval/wiki_eval/datasets/frames.jsonl`

- [ ] **Step 4: Sanity-check the output shape**

Run: `head -n 1 wiki_eval/datasets/frames.jsonl | python -c "import sys, json; r=json.loads(sys.stdin.read()); print(sorted(r)); print(type(r['reference_pages']), r['reference_pages'][:1])"`
Expected: keys `['input', 'reference_pages', 'target']`; `reference_pages` is a `list` whose first item is an `https://en.wikipedia.org/wiki/...` URL.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_frames.py wiki_eval/datasets/frames.jsonl pyproject.toml uv.lock
git commit -m "Add FRAMES dataset converter and 100-row committed subsample

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `frames()` task wiring

**Files:**
- Modify: `eval/wiki_eval/tasks.py` (add `record_to_sample` + `frames` task)
- Test: `eval/tests/test_tasks.py` (create)

**Interfaces:**
- Consumes: `retrieval_grounding`, `correctness_judge`, `used_wikipedia_tool` (Tasks 1–2 / existing); `frames.jsonl` (Task 3).
- Produces: `_frames_record_to_sample(record: dict) -> Sample`; `frames()` `@task`.

- [ ] **Step 1: Write the failing test**

Create `eval/tests/test_tasks.py`:

```python
"""Unit test for the FRAMES record->Sample mapping (no API/network)."""

from wiki_eval.tasks import _frames_record_to_sample


def test_frames_record_maps_reference_pages_into_metadata():
    record = {
        "input": "Who walked on the Moon first?",
        "target": "Neil Armstrong",
        "reference_pages": ["https://en.wikipedia.org/wiki/Apollo_11"],
    }
    sample = _frames_record_to_sample(record)
    assert sample.input == "Who walked on the Moon first?"
    assert sample.target == "Neil Armstrong"
    assert sample.metadata["reference_pages"] == ["https://en.wikipedia.org/wiki/Apollo_11"]


def test_frames_record_defaults_missing_reference_pages_to_empty():
    sample = _frames_record_to_sample({"input": "Q", "target": "A"})
    assert sample.metadata["reference_pages"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tasks.py -v`
Expected: FAIL — `ImportError: cannot import name '_frames_record_to_sample'`.

- [ ] **Step 3: Wire the task**

In `eval/wiki_eval/tasks.py`, update the dataset import to also bring in `Sample`. Change:

```python
from inspect_ai.dataset import json_dataset
```

to:

```python
from inspect_ai.dataset import Sample, json_dataset
```

Update the scorers import. Change:

```python
from wiki_eval.scorers import correctness_judge, used_wikipedia_tool
```

to:

```python
from wiki_eval.scorers import correctness_judge, retrieval_grounding, used_wikipedia_tool
```

Add, after the existing `factual_qa` task:

```python
def _frames_record_to_sample(record: dict) -> Sample:
    """Map a FRAMES jsonl row to a Sample, carrying gold pages into metadata.

    `reference_pages` is read only by the retrieval_grounding scorer, which is
    why grounding stays FRAMES-only: no other dataset sets this key.
    """
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={"reference_pages": record.get("reference_pages", [])},
    )


@task
def frames():
    """FRAMES multi-hop Wikipedia QA, with a retrieval-grounding (recall) signal."""
    return Task(
        dataset=json_dataset(
            str(_DATASETS / "frames.jsonl"),
            sample_fields=_frames_record_to_sample,
        ),
        solver=wiki_agent_solver(),
        scorer=[correctness_judge(), used_wikipedia_tool(), retrieval_grounding()],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tasks.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Run the full offline suite**

Run: `uv run pytest`
Expected: PASS — all tests (existing scorer/solver tests + new grounding + tasks tests), no `ANTHROPIC_API_KEY` required.

- [ ] **Step 6: Commit**

```bash
git add wiki_eval/tasks.py tests/test_tasks.py
git commit -m "Add FRAMES task wiring grounding scorer to gold reference pages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Live smoke test + reporting verification (manual)

> This task needs `ANTHROPIC_API_KEY` and network. It is a manual verification of the dict-metric reporting, not an automated test. Do it once after Task 4.

**Files:** none (verification only).

- [ ] **Step 1: Run a tiny live eval**

Run: `uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5 --limit 2`
Expected: completes; the results summary lists `correctness_judge` accuracy plus `retrieval_grounding` with **recall, precision, and f1** rows.

- [ ] **Step 2: Confirm the dict-metric form rendered**

If the results table shows `recall`, `precision`, and `f1`, the design's primary reporting form works — done.

**Fallback (only if Inspect rejected the dict-shaped `metrics`):** change the decorator to `@scorer(metrics=[mean(), stderr()])`, return `Score(value=values["recall"], ...)`, and move `precision`/`f1` into the `metadata` dict. Re-run Step 1. Then commit:

```bash
git add wiki_eval/scorers.py
git commit -m "Report grounding recall as headline metric; precision/F1 in metadata

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: (Optional) View the run**

Run: `uv run inspect view start --host 0.0.0.0 --port 7575`
Expected: per-sample grounding values and explanations visible in the transcript.

---

## Follow-ups (out of scope here)

- `max_steps` on the FRAMES solver stays at the default (6). FRAMES questions need 2–15 hops, so this likely floors recall — tune it as a separate calibration task once baseline numbers exist.
- Per-`reasoning_types` accuracy breakdown is deferred (the field is available in FRAMES if we choose to carry it later).
