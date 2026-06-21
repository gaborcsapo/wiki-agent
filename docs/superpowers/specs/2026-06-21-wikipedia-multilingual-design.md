# Wikipedia tool: multilingual support + iterative climb on the multilingual benchmark

**Date:** 2026-06-21
**Scope:** feature in `agent/`; experiment tooling in `eval/`. No change to the
agent↔eval boundary, `AgentResult`, or `run()`.

## Problem & goal

The tool only queries `en.wikipedia.org`, but the `multilingual_qa` benchmark (30
samples) asks English questions whose facts live on native-language Wikipedias
(Hungarian, Icelandic, Estonian, Swahili, Armenian, Welsh, Basque, Georgian,
Yoruba). **Goal: raise correctness on `multilingual_qa`** via 3 improve→measure
cycles, fixing the failure modes surfaced each cycle.

The judge is language-neutral (a correct fact in any language scores), and
correctness is cache-independent — so the experiment needs only **one run per
cycle** (no warm-up protocol).

## Experiment protocol

- Benchmark: `multilingual_qa`, all **30** samples.
- Agent: **Haiku** (`claude-haiku-4-5`), pinned in-process via `run_pinned.py`
  (immune to concurrent edits of the shared `AGENT_MODEL`).
- One run each: **baseline → cycle 1 → cycle 2 → cycle 3** (4 runs total).
- Clear the cache once at the start (correctness is cache-independent; this just
  avoids stale-key confusion when the cache key gains a language component).
- After each run: record overall accuracy + per-category + per-language
  breakdown, and **investigate failures** (which languages/categories miss, and
  what `lang` the agent actually queried) to choose the next change.

Run command: `cd eval && uv run python run_pinned.py multilingual_qa claude-haiku-4-5 30`.
Analysis: `uv run python -m wiki_eval.analyze_multilingual <log.eval>`.

## Cycle 1 (planned): query any-language Wikipedia

The dominant lever. Add a `lang` parameter end-to-end:

- `config`: `WIKI_API_TEMPLATE = "https://{lang}.wikipedia.org/w/api.php"`,
  `DEFAULT_LANG = "en"`; `WIKI_API` kept as the en default for back-compat.
- `wikipedia._api_url(lang)`; `_get(params, client, lang=DEFAULT_LANG)` builds the
  per-language URL **and includes `lang` in the cache key** (else `es`/`en` with
  identical params would collide).
- `search`/`get_article`/`search_many`/`get_articles` gain `lang=DEFAULT_LANG`;
  `dispatch` reads `lang` from the tool input.
- Schema: add a `lang` string field (default "en"; e.g. hu, is, et, sw, hy, cy,
  eu, ka, yo, de, fr, es, ja).
- Prompt: tell the agent Wikipedia has per-language editions and to query the
  edition of the language/country the question is about (with code examples),
  especially for local facts and native-language questions.

## Cycles 2 & 3 (failure-driven, decided after each run)

Chosen from the failures the prior cycle surfaces. Likely candidates:
- **Interlanguage links (`langlinks`)** to pivot a title between English and the
  native edition (find the native title from an English hit, or vice versa).
- **Cross-edition fallback**: when a native search/title yields nothing, retry on
  English (or vice versa) — at the tool or prompt level.
- **Language-code guidance / auto-pivot** improvements (e.g. returning available
  langlinks in `get_article` so the agent can navigate).

Each cycle = one tool change + one rerun; the actual change is recorded in the
report once failures are known.

## Tests (offline, no network)

- `_api_url(lang)` builds the right host; defaults to en.
- `_get` includes `lang` in the cache key (en vs es with same params don't
  collide) — via fakes/`tmp_path`.
- `search`/`get_article`/dispatch thread `lang` through (monkeypatched singles).
- Any cycle-2/3 pure helpers (e.g. langlinks parsing) unit-tested.

## Analysis tooling

`eval/wiki_eval/analyze_multilingual.py`: reads one `.eval` log; prints overall
accuracy, per-category, per-language, and a failure list (question + gold +
the `lang` values the agent used). Pure `_summarize(records)` unit-tested.

## Deliverable

A very concise report: the per-cycle accuracy curve, which change moved it most,
and the final state with remaining failure modes.

## Files touched

| Path | Change |
|------|--------|
| `agent/wiki_agent/config.py` | `WIKI_API_TEMPLATE`, `DEFAULT_LANG` |
| `agent/wiki_agent/wikipedia.py` | `lang` end-to-end; cache key includes lang; cycle 2/3 changes |
| `agent/wiki_agent/agent.py` | prompt guidance (cycle 1; refined in 2/3 if needed) |
| `agent/tests/test_wikipedia.py` | lang routing + cache-key tests |
| `eval/wiki_eval/analyze_multilingual.py` + test | per-run breakdown + failures |
| `eval/run_pinned.py` | accept a task name argument |
| `docs/superpowers/reports/2026-06-21-multilingual-climb.md` | the report |
