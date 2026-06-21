# FRAMES retrieval-grounding scorer — design

**Date:** 2026-06-21
**Status:** Approved (design)
**Scope:** `eval/` only. The agent is not modified.

## Goal

Add the FRAMES benchmark to the eval suite and introduce one new, FRAMES-only
signal: **retrieval grounding** — how well the pages the agent actually read
overlap with FRAMES's gold set of reference Wikipedia pages for each question.

FRAMES is hard and headroom is large, so it becomes the primary benchmark for
iterating on the agent loop and calibrating prompts. Correctness stays the
primary metric; grounding is a diagnostic that separates *retrieval* failure
("never read the right page") from *reasoning* failure ("read it, still wrong").

Small/custom benchmarks must stay simple: correctness only. Grounding must not
leak into them.

## Guiding principle: FRAMES-only by composition, not by flags

No conditional logic anywhere. The grounding scorer reads its gold pages from
`state.metadata["reference_pages"]`, which **only the FRAMES dataset loader
populates**. The scorer is FRAMES-only purely because only the `frames()` task
lists it in its `scorer=[...]`. `factual_qa()` and future small benchmarks keep
`scorer=[correctness_judge()]` (plus existing `used_wikipedia_tool()`) and never
touch grounding. No new cross-project coupling; the agent is untouched.

## Data shape (FRAMES, `google/frames-benchmark`)

- 824 rows, single `test` split.
- Relevant fields: `Prompt` (question), `Answer` (gold, usually terse),
  `wiki_links` (a stringified Python list of gold reference Wikipedia URLs),
  `reasoning_types` (labels). The exploded `wikipedia_link_1..N` columns are
  redundant with `wiki_links` and are ignored.

## Page-matching method

- **Gold set:** each URL in the sample's `reference_pages` → normalized slug.
- **Read set:** scan trajectory `tool_result` steps and regex-extract every
  `https://en.wikipedia.org/wiki/<slug>`. Only `get_article` results contain
  that canonical URL line (plaintext intro extracts and search snippets do not),
  so there are no false matches. Using the **resolved** URL from the result —
  not the title the agent typed — means redirects and case differences
  (e.g. `USA` → `United States`) match the gold automatically.
- **Normalization** (`_normalize_wiki_url`, pure): take the path after `/wiki/`,
  strip query/fragment, percent-decode, `_` → space, casefold, strip. Applied to
  both gold and read sets before comparison. Returns `None` for non-article URLs.

## The scorer: `retrieval_grounding()`

Pure, unit-testable helpers (no network, no API key):

- `_normalize_wiki_url(url) -> str | None`
- `_fetched_pages(steps) -> set[str]` — resolved slugs read via `get_article`.
- `_grounding_scores(gold: set, read: set) -> {recall, precision, f1}`

Metric semantics (decided): **recall is the headline; precision and F1 ride
along as diagnostics.**

- `recall = |read ∩ gold| / |gold|` — did the agent find the needed evidence?
- `precision = |read ∩ gold| / |read|` — diagnostic; over-exploration is not a
  failure we optimize against.
- `f1` — harmonic mean, diagnostic.

The `@scorer` reads `reference_pages` from sample metadata and `trajectory.steps`
from the solver's metadata, then returns:

```python
Score(
    value={"recall": r, "precision": p, "f1": f1},
    explanation=f"{hits}/{len(gold)} gold pages read; {len(read)} read total",
    metadata={"n_gold": ..., "n_read": ..., "n_hit": ...},
)
```

Empty/missing gold (should not occur on real FRAMES rows) → defensive
`recall = 0` with an explanatory note, rather than raising into the run.

## Dataset & task wiring

- **`eval/wiki_eval/datasets/frames.jsonl`** — extends the JSONL schema *for this
  benchmark only* with a third key:
  `{"input": Prompt, "target": Answer, "reference_pages": [urls...]}`.
  A **subsampled ~100-row file is committed** so the eval is self-contained and
  tests run offline. (Full 824 can be regenerated; see script.)
- **`eval/scripts/build_frames.py`** — dev-only, not imported by the package and
  not covered by tests. Downloads `google/frames-benchmark`, parses the
  `wiki_links` string with `ast.literal_eval` into a list, writes the jsonl with
  a `--limit` flag for subsampling. Documents provenance and lets us refresh.
- **`eval/wiki_eval/tasks.py`** — new `frames()` task using
  `json_dataset(path, sample_fields=record_to_sample)`, where a small
  `record_to_sample` maps `input`/`target` and puts `reference_pages` into
  `Sample.metadata`. `factual_qa()` stays on the plain 2-field `json_dataset`,
  unchanged.

## Reporting in the suite

Dict-valued `Score` with
`@scorer(metrics={"recall": [mean(), stderr()], "precision": [mean()], "f1": [mean()]})`.
The CLI summary and `inspect view` then show `retrieval_grounding/recall`,
`/precision`, `/f1` as their own rows next to `correctness_judge` accuracy —
recall headline, precision/F1 as diagnostics.

API detail to confirm at build time: Inspect's exact dict-metric syntax. If it
differs, fall back to `value = recall` (single float) with precision/F1 carried
in `Score.metadata`.

## Testing

New `eval/tests/test_grounding.py`, all offline (must pass with no
`ANTHROPIC_API_KEY`):

- `_normalize_wiki_url`: redirects, percent-encoding, underscores vs spaces,
  query/fragment stripping, non-article URLs → `None`.
- `_fetched_pages`: ignores search-result and error `tool_result` steps; collects
  resolved `get_article` URLs; dedupes.
- `_grounding_scores`: perfect overlap, partial, zero, and over-exploration
  (extra reads lower precision but not recall).

## Files touched

| Path | Change |
|------|--------|
| `eval/wiki_eval/scorers.py` | + `retrieval_grounding()` scorer and pure helpers |
| `eval/wiki_eval/tasks.py` | + `frames()` task and `record_to_sample` |
| `eval/wiki_eval/datasets/frames.jsonl` | new (committed ~100-row subsample) |
| `eval/scripts/build_frames.py` | new (dev-only converter) |
| `eval/tests/test_grounding.py` | new |

**No changes to the agent.** The one allowed coupling (`eval` → `wiki_agent.run`)
is unchanged.

## Out of scope (YAGNI)

- Per-reasoning-type accuracy breakdown (possible later via `reasoning_types`).
- Retrieval-adequacy / hop-count efficiency metric.
- Adding grounding to any non-FRAMES benchmark.
- Changing `max_steps` (tracked separately as a calibration question).
