# Wikipedia tool: rate-limit hardening, backoff, cache & empirical benchmark

**Date:** 2026-06-21
**Scope:** `agent/` only. No changes to `eval/`. The one allowed coupling
(`eval → wiki_agent.run`) is untouched; nothing here changes `AgentResult` or the
`run()` signature.

## Problem

The Wikipedia tool gets rate-limited when many calls are made (e.g. eval runs).
It has no retry/backoff and no caching, so transient throttling fails a request
outright and re-running a benchmark re-fetches every page. We want:

1. The best **anonymous** setup for the highest sustained request rate.
2. **Backoff** that respects Wikimedia's signals.
3. A **simple, isolated, cleanly-managed cache** so benchmark re-runs reuse pages.
4. **Empirical testing** of different setups to pick the best, run once now.

## Research findings (English Wikipedia Action API, 2026)

Authoritative sources: mediawiki.org `Wikimedia_APIs/Rate_limits`, `API:Etiquette`,
`Manual:Maxlag_parameter`, `API:Query`, `API:Data_formats`; Foundation User-Agent
Policy. Key points that drive the design:

- **User-Agent is the dominant lever.** A compliant UA → **200 req/min** tier;
  a generic/empty UA → **10 req/min** (and risks a 403 / silent IP block).
  Recommended format includes the library/version, e.g.
  `wiki-agent/… (url; email) python-httpx/0.28.1`.
- **Auth does not help reads** under the 2026 rules (a compliant anonymous client
  already gets 200/min; higher tiers need established-editor or bot-flag status).
  → We stay **anonymous** (user's decision).
- **Serial requests** are recommended; concurrency ceiling is **3**. The agent is
  serial by construction (one tool call at a time), so this is already satisfied.
- **`maxlag=5`** for non-interactive/batch traffic (sheds load under DB lag).
- **gzip** (`Accept-Encoding: gzip`) recommended — already set; httpx decompresses.
- **Backoff:** on **429 / 503**, honor `Retry-After`; if absent, wait **≥5 s then
  exponential**. The Action API can also return a **`maxlag` error inside an HTTP
  200 body** (`{"error":{"code":"maxlag"}}`) — must be treated as retryable.
  Do **not** rely on undocumented `X-RateLimit-*` headers.

## Design

All work is in `agent/wiki_agent/`. Pure logic stays separated from I/O and is
unit-tested with fakes/monkeypatch — **no live network in the test suite** (the
live benchmark is a separate, opt-in script).

### 1. Config (`config.py`)

New constants (single source of truth):

```python
# MediaWiki throughput / etiquette
MAXLAG = 5                 # shed load under replica lag (non-interactive)

# Backoff (Wikimedia: honor Retry-After, else >=5s then exponential)
MAX_RETRIES   = 4
BACKOFF_BASE  = 1.0        # seconds, exponential base
BACKOFF_CAP   = 30.0       # seconds, max single wait
MIN_RETRY_WAIT = 5.0       # Wikimedia floor when no Retry-After header

# Disk cache (isolated, no eviction — simplicity over size management)
CACHE_ENABLED = True
CACHE_DIR = Path(__file__).resolve().parent.parent / ".wiki_cache"
```

`USER_AGENT` gains the library/version suffix (`python-httpx/<version>`).

### 2. Cache (`wiki_agent/cache.py`) — new, tiny module

Disk cache of **raw API JSON**, keyed by the *semantic* request params. One
responsibility: persist/retrieve a dict by a stable key.

- `_key(params: dict) -> str` — `sha256(json.dumps(params, sort_keys=True))`.
  Pure, key is order-independent. (unit-tested)
- `get(params) -> dict | None` — returns cached JSON or `None` (miss / disabled /
  unreadable file). Reads `config.CACHE_DIR` **dynamically** so tests can point it
  at `tmp_path`.
- `set(params, data) -> None` — writes `<CACHE_DIR>/<key>.json`, creating the dir.
- `clear() -> int` — deletes all `*.json` in the dir, returns the count.

Honors `config.CACHE_ENABLED` (get/set are no-ops when disabled). No TTL, no
eviction (per "don't worry about size"). Errors never raise into callers.

### 3. Backoff + cache integration (`wikipedia._get`)

`_get(params, client)` becomes the single integration point. Pure helpers added:

- `_retry_delay(attempt, retry_after) -> float` — if `retry_after` parses as a
  number, return it; else `min(CAP, max(MIN_RETRY_WAIT, BASE * 2**attempt))`.
  (unit-tested, no clock)
- `_is_maxlag(data) -> bool` — `data.get("error", {}).get("code") == "maxlag"`.
  (unit-tested)

Flow:

1. Build the cache key from the caller's semantic `params` (excludes the transport
   base: `format`, `formatversion`, `maxlag`) so hits are stable across runs.
2. `cached = cache.get(params)`; if hit, return it (no HTTP).
3. Otherwise loop up to `MAX_RETRIES + 1`:
   - `resp = client.get(WIKI_API, params={**base, **params})` where `base` now
     includes `maxlag=config.MAXLAG`.
   - On **429/503**: if attempts remain, `_sleep(_retry_delay(attempt, Retry-After))`
     and retry; else `resp.raise_for_status()` (becomes the existing readable
     error string in `search`/`get_article`).
   - Else `resp.raise_for_status()`, `data = resp.json()`.
   - If `_is_maxlag(data)`: same backoff-and-retry as above (final attempt returns
     the body, which the parser renders as a benign "no extract" message).
   - Else **success**: `cache.set(params, data)` (only non-error bodies) and return.

Sleeping goes through a module-level indirection (`_sleep = time.sleep`) that tests
monkeypatch — **no real sleeps in tests**. The public `search`/`get_article`/
`dispatch` signatures are unchanged; their `except httpx.HTTPError` still yields a
readable string after retries are exhausted.

### 4. CLI knobs (`cli.py`, `ask` command)

- `--no-cache` → sets `config.CACHE_ENABLED = False` for the run.
- `--clear-cache` → calls `cache.clear()` and reports the count before running.

Simple runtime toggles; no plumbing through `agent.run`.

### 5. Empirical benchmark (`wiki_agent/ratelimit_bench.py`) — new, opt-in

`python -m wiki_agent.ratelimit_bench`. Live, **cache-bypassed**, run once now.

- **Setups (4):** `{serial (conc=1), conc=3} × {maxlag off, maxlag=5}`, all with the
  compliant UA + gzip. (No empty/generic-UA testing — block risk; that effect is
  taken from the docs.)
- **Per setup (≤500-request hard budget):** fire in **2-second probe windows**
  (push as fast as the setup allows for 2 s, brief pause, repeat) until 500 sent
  **or** early-stop. **Early-stop:** ≥10 consecutive 429/503 → record & stop.
- **Between setups:** **60 s** back-off so the per-minute window resets and one
  setup can't bias the next.
- **Workload:** one fixed, varied list of real queries (mix of `search` +
  `get_article`, diverse titles), cycled identically across setups.
- **Metrics per setup:** requests sent, successes, 429 / 503 / maxlag-body counts,
  timeouts, latency p50/p95, wall-clock, **effective successful req/s**,
  **first-throttle index**.
- **Selection:** pure `_summarize(results)` and `_pick_best(summaries)` (rank by
  highest sustained successful req/s, fewest throttle events). The runner prints a
  comparison table and the recommended config; pure helpers are unit-tested.
- **Footprint:** ≤2,000 requests over ~5–7 min including cooldowns — bounded.

### 6. Tests (all offline, no API key)

`tests/test_wikipedia.py` (extend) + `tests/test_cache.py` + bench helper tests:

- `_retry_delay`: numeric `Retry-After` honored; exponential with floor/cap when
  absent.
- `_is_maxlag`: true on maxlag body, false otherwise.
- Retry loop via a **fake client**: 429→200 returns data after one sleep;
  maxlag-body→200 likewise; exhaustion raises → readable error string. Sleeps are
  recorded, never real.
- Cache: `set`→`get` round-trip (`tmp_path` via monkeypatched `CACHE_DIR`); miss →
  `None`; disabled → `None`; key order-independence; `clear()` count.
- `_get` cache hit: second identical call serves from cache with **0** client calls.
- Bench: `_summarize` / `_pick_best` on synthetic results.

## Non-goals

- No authentication / OAuth (no read benefit; user chose anonymous).
- No cache eviction, TTL, or size cap (explicitly out of scope).
- No concurrency added to the **agent** itself (it stays serial; concurrency is
  only exercised inside the benchmark for comparison).
- No changes to `eval/` or the agent↔eval boundary.

## Files touched

| Path | Change |
|------|--------|
| `agent/wiki_agent/config.py` | New backoff/cache/maxlag constants; UA suffix |
| `agent/wiki_agent/cache.py` | **New** — tiny disk cache |
| `agent/wiki_agent/wikipedia.py` | Backoff + cache in `_get`; pure helpers |
| `agent/wiki_agent/cli.py` | `--no-cache` / `--clear-cache` on `ask` |
| `agent/wiki_agent/ratelimit_bench.py` | **New** — live benchmark + pure scoring |
| `agent/tests/test_wikipedia.py` | Retry/maxlag/cache-hit tests |
| `agent/tests/test_cache.py` | **New** — cache unit tests |
| `agent/tests/test_ratelimit_bench.py` | **New** — `_summarize`/`_pick_best` tests |
| `.gitignore` | Add `agent/.wiki_cache/` |
| `CLAUDE.md` | Document cache, backoff, bench command |
