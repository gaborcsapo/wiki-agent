# Wikipedia Tool Rate-Limit Hardening, Backoff & Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the agent's Wikipedia tool for high-throughput anonymous use: compliant headers + `maxlag`, Retry-After/exponential backoff, an isolated disk cache for benchmark re-runs, and a one-shot live benchmark that picks the best setup.

**Architecture:** All changes are confined to `agent/wiki_agent/`. The single HTTP chokepoint `wikipedia._get` gains a disk cache (raw JSON keyed by semantic params) wrapped around a retry loop. Pure helpers (`_retry_delay`, `_is_maxlag`, cache `_key`, bench `_summarize`/`_pick_best`) are separated from I/O and unit-tested with fakes — no live network in the suite. A standalone `ratelimit_bench` module does the live, opt-in measurement.

**Tech Stack:** Python 3.12, `httpx` 0.28.1, `click`, `rich`, `pytest`. Stdlib `hashlib`/`json`/`time`/`concurrent.futures`.

## Global Constraints

- Scope is `agent/` only. **Never** import from `eval/`; do not change `AgentResult` or the `run()` signature.
- Anonymous only — no auth/OAuth.
- Keep pure logic separate from I/O; unit-test all custom logic with fakes/monkeypatch. **No live network or API key in the test suite.**
- All config constants live in `agent/wiki_agent/config.py` — no hardcoded values elsewhere.
- The tool's public functions return readable strings and never raise into the model.
- Git: stage only the paths you changed (no `git add -A`); commit per task; messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Config constants + compliant User-Agent suffix

**Files:**
- Modify: `agent/wiki_agent/config.py`

**Interfaces:**
- Produces: `config.MAXLAG:int=5`, `config.MAX_RETRIES:int=4`, `config.BACKOFF_BASE:float=1.0`, `config.BACKOFF_CAP:float=30.0`, `config.MIN_RETRY_WAIT:float=5.0`, `config.CACHE_ENABLED:bool=True`, `config.CACHE_DIR:Path` (= `agent/.wiki_cache`). `config.USER_AGENT` gains a `python-httpx/<version>` suffix.

- [ ] **Step 1: Add imports and constants to `config.py`**

At the top, add `from pathlib import Path` and `import httpx`. Replace the `USER_AGENT` suffix `httpx` with the library/version form, and append the new constants:

```python
from pathlib import Path

import httpx

# ... existing AGENT_MODEL, MAX_TOKENS, DEFAULT_MAX_STEPS ...

# MediaWiki API.
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia policy requires a descriptive User-Agent with contact info and the
# underlying library/version. A compliant UA grants the 200 req/min tier
# (vs 10 req/min for a generic one).
USER_AGENT = (
    "WikiAgent/0.1 (https://github.com/gaborxcsapo/anthropic-takehome; "
    f"gaborxcsapo@gmail.com) python-httpx/{httpx.__version__}"
)
HTTP_TIMEOUT = 15.0

# MediaWiki etiquette: shed load under DB replica lag on non-interactive traffic.
MAXLAG = 5

# Backoff: Wikimedia asks clients to honor Retry-After, else wait >=5s then
# back off exponentially. Applied in wikipedia._get.
MAX_RETRIES = 4
BACKOFF_BASE = 1.0       # seconds (exponential base)
BACKOFF_CAP = 30.0       # seconds (max single wait)
MIN_RETRY_WAIT = 5.0     # seconds (floor when no Retry-After header)

# Disk cache for raw API JSON. Isolated dir, no eviction/TTL (simplicity over
# size). agent/.wiki_cache (parent.parent of this file = the agent/ root).
CACHE_ENABLED = True
CACHE_DIR = Path(__file__).resolve().parent.parent / ".wiki_cache"
```

(Keep the existing `DEFAULT_SEARCH_LIMIT`/`DEFAULT_EXTRACT_CHARS`.)

- [ ] **Step 2: Verify it imports**

Run: `cd agent && uv run python -c "from wiki_agent import config; print(config.USER_AGENT); print(config.CACHE_DIR)"`
Expected: UA string ending `python-httpx/0.28.1` and a path ending `/agent/.wiki_cache`.

- [ ] **Step 3: Commit**

```bash
git add agent/wiki_agent/config.py
git commit -m "Add maxlag/backoff/cache config + library version in User-Agent"
```

---

### Task 2: Isolated disk cache module

**Files:**
- Create: `agent/wiki_agent/cache.py`
- Create: `agent/tests/test_cache.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `config.CACHE_ENABLED`, `config.CACHE_DIR` (read dynamically).
- Produces: `cache._key(params:dict)->str`, `cache.get(params:dict)->dict|None`, `cache.set(params:dict, data:dict)->None`, `cache.clear()->int`.

- [ ] **Step 1: Write the failing tests** in `agent/tests/test_cache.py`

```python
"""Unit tests for the isolated disk cache — no network, tmp_path only."""

import pytest

from wiki_agent import cache, config


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    return tmp_path


def test_set_then_get_roundtrip():
    params = {"action": "query", "titles": "Cat"}
    cache.set(params, {"x": 1})
    assert cache.get(params) == {"x": 1}


def test_get_miss_returns_none():
    assert cache.get({"action": "nope"}) is None


def test_disabled_get_and_set_are_noops(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    cache.set({"a": 1}, {"v": 1})
    assert cache.get({"a": 1}) is None


def test_key_is_order_independent():
    assert cache._key({"a": 1, "b": 2}) == cache._key({"b": 2, "a": 1})


def test_clear_removes_entries_and_counts():
    cache.set({"a": 1}, {"v": 1})
    cache.set({"a": 2}, {"v": 2})
    assert cache.clear() == 2
    assert cache.get({"a": 1}) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd agent && uv run pytest tests/test_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: ... cache` / attribute errors.

- [ ] **Step 3: Implement `agent/wiki_agent/cache.py`**

```python
"""A tiny, isolated disk cache for raw MediaWiki API JSON responses.

Keyed by the semantic request params so identical lookups across benchmark
re-runs are served from disk instead of re-fetched. Deliberately simple: no
TTL, no eviction, no size management. Never raises into callers.
"""

from __future__ import annotations

import hashlib
import json

from . import config


def _key(params: dict) -> str:
    """Stable, order-independent cache key for a set of request params."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(params: dict) -> dict | None:
    """Return cached JSON for ``params``, or None on miss/disabled/unreadable."""
    if not config.CACHE_ENABLED:
        return None
    path = config.CACHE_DIR / f"{_key(params)}.json"
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def set(params: dict, data: dict) -> None:
    """Persist ``data`` for ``params``. No-op when caching is disabled."""
    if not config.CACHE_ENABLED:
        return
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / f"{_key(params)}.json"
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except OSError:
        pass


def clear() -> int:
    """Delete all cached entries; return how many files were removed."""
    if not config.CACHE_DIR.exists():
        return 0
    count = 0
    for path in config.CACHE_DIR.glob("*.json"):
        try:
            path.unlink()
            count += 1
        except OSError:
            pass
    return count
```

- [ ] **Step 4: Run to verify pass**

Run: `cd agent && uv run pytest tests/test_cache.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Gitignore the cache dir**

Add under the "Agent run artifacts" section of the root `.gitignore`:

```
agent/.wiki_cache/
```

- [ ] **Step 6: Commit**

```bash
git add agent/wiki_agent/cache.py agent/tests/test_cache.py .gitignore
git commit -m "Add isolated disk cache for Wikipedia API JSON"
```

---

### Task 3: Backoff + cache integration in `_get`

**Files:**
- Modify: `agent/wiki_agent/wikipedia.py`
- Modify: `agent/tests/test_wikipedia.py`

**Interfaces:**
- Consumes: `config.MAX_RETRIES/BACKOFF_BASE/BACKOFF_CAP/MIN_RETRY_WAIT/MAXLAG`, `cache.get/set`.
- Produces: `wikipedia._retry_delay(attempt:int, retry_after:str|None)->float`, `wikipedia._is_maxlag(data:dict)->bool`, `wikipedia._sleep(seconds:float)->None` (monkeypatch point). `_get` signature unchanged: `_get(params:dict, client:httpx.Client)->dict`.

- [ ] **Step 1: Write the failing tests** — append to `agent/tests/test_wikipedia.py`

Add imports at the top of the file: `import httpx`, `import pytest`, and extend the existing `from wiki_agent import wikipedia` with `from wiki_agent import config`. Then append:

```python
class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"HTTP {self.status_code}")


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None):
        self.calls += 1
        return self._responses.pop(0)


def test_retry_delay_honors_numeric_retry_after():
    assert wikipedia._retry_delay(0, "12") == 12.0


def test_retry_delay_exponential_floor_and_cap():
    assert wikipedia._retry_delay(0, None) == config.MIN_RETRY_WAIT
    assert wikipedia._retry_delay(10, None) == config.BACKOFF_CAP


def test_is_maxlag():
    assert wikipedia._is_maxlag({"error": {"code": "maxlag"}}) is True
    assert wikipedia._is_maxlag({"query": {}}) is False


def test_get_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    slept = []
    monkeypatch.setattr(wikipedia, "_sleep", lambda s: slept.append(s))
    client = FakeClient([
        FakeResponse(429, headers={"Retry-After": "3"}),
        FakeResponse(200, {"query": {"ok": True}}),
    ])
    data = wikipedia._get({"action": "query"}, client)
    assert data == {"query": {"ok": True}}
    assert slept == [3.0]
    assert client.calls == 2


def test_get_retries_on_maxlag_body(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    monkeypatch.setattr(wikipedia, "_sleep", lambda s: None)
    client = FakeClient([
        FakeResponse(200, {"error": {"code": "maxlag"}}, {"Retry-After": "5"}),
        FakeResponse(200, {"query": {"ok": True}}),
    ])
    data = wikipedia._get({"action": "query"}, client)
    assert data == {"query": {"ok": True}}
    assert client.calls == 2


def test_get_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    monkeypatch.setattr(wikipedia, "_sleep", lambda s: None)
    client = FakeClient([FakeResponse(503) for _ in range(config.MAX_RETRIES + 1)])
    with pytest.raises(httpx.HTTPError):
        wikipedia._get({"action": "query"}, client)


def test_get_served_from_cache_skips_http(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    client = FakeClient([FakeResponse(200, {"query": {"v": 1}})])
    first = wikipedia._get({"action": "query", "titles": "Cat"}, client)
    assert first == {"query": {"v": 1}} and client.calls == 1
    second = wikipedia._get({"action": "query", "titles": "Cat"}, FakeClient([]))
    assert second == {"query": {"v": 1}}  # no pop from empty -> served from cache
```

- [ ] **Step 2: Run to verify failure**

Run: `cd agent && uv run pytest tests/test_wikipedia.py -q`
Expected: FAIL — `_retry_delay`/`_is_maxlag`/`_sleep` undefined; `_get` lacks cache/retry.

- [ ] **Step 3: Implement in `agent/wiki_agent/wikipedia.py`**

Add `import time` and `from . import cache` (alongside `from . import config`). Replace the existing `_get` and add the helpers in the "HTTP I/O" section:

```python
_RETRY_STATUS = {429, 503}


def _sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch out real sleeping."""
    time.sleep(seconds)


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before a retry.

    Honor a numeric ``Retry-After`` header; otherwise exponential backoff
    floored at MIN_RETRY_WAIT and capped at BACKOFF_CAP.
    """
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    backoff = config.BACKOFF_BASE * (2 ** attempt)
    return min(config.BACKOFF_CAP, max(config.MIN_RETRY_WAIT, backoff))


def _is_maxlag(data: dict) -> bool:
    """True if a 200 body is actually a maxlag rejection."""
    return data.get("error", {}).get("code") == "maxlag"


def _get(params: dict, client: httpx.Client) -> dict:
    """MediaWiki API GET with disk cache + Retry-After/exponential backoff.

    ``params`` are the semantic query params; transport params (format,
    formatversion, maxlag) are added here and excluded from the cache key so
    cache hits stay stable across runs.
    """
    cached = cache.get(params)
    if cached is not None:
        return cached

    base = {"format": "json", "formatversion": 2, "maxlag": config.MAXLAG}
    for attempt in range(config.MAX_RETRIES + 1):
        response = client.get(config.WIKI_API, params={**base, **params})
        if response.status_code in _RETRY_STATUS:
            if attempt < config.MAX_RETRIES:
                _sleep(_retry_delay(attempt, response.headers.get("Retry-After")))
                continue
            response.raise_for_status()  # exhausted -> raise to the caller
        response.raise_for_status()
        data = response.json()
        if _is_maxlag(data):
            if attempt < config.MAX_RETRIES:
                _sleep(_retry_delay(attempt, response.headers.get("Retry-After")))
                continue
            return data  # exhausted maxlag -> parser renders a benign message
        cache.set(params, data)
        return data

    raise httpx.HTTPError("retries exhausted")  # pragma: no cover
```

Leave `_make_client`, `search`, `get_article`, `dispatch`, and the parsers unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `cd agent && uv run pytest tests/test_wikipedia.py -q`
Expected: PASS (existing tests + 7 new).

- [ ] **Step 5: Commit**

```bash
git add agent/wiki_agent/wikipedia.py agent/tests/test_wikipedia.py
git commit -m "Add Retry-After/exponential backoff + cache to _get"
```

---

### Task 4: CLI cache toggles

**Files:**
- Modify: `agent/wiki_agent/cli.py`

**Interfaces:**
- Consumes: `cache.clear()`, `config.CACHE_ENABLED`.
- Produces: `--no-cache` and `--clear-cache` flags on the `ask` command.

- [ ] **Step 1: Add the import**

In `cli.py` add `cache` to the package import: `from . import cache, config` (it currently imports `config`).

- [ ] **Step 2: Add the options and handling to `ask`**

Add two options above `def ask(...)` and widen its signature, then handle them first in the body:

```python
@click.option("--no-cache", is_flag=True, help="Bypass the Wikipedia disk cache for this run.")
@click.option("--clear-cache", is_flag=True, help="Delete cached Wikipedia pages before running.")
def ask(question: str, model: str | None, max_steps: int, save: bool,
        no_cache: bool, clear_cache: bool) -> None:
    if clear_cache:
        removed = cache.clear()
        console.print(f"[dim]Cleared {removed} cached Wikipedia entries.[/dim]")
    if no_cache:
        config.CACHE_ENABLED = False
    # ... existing body unchanged ...
```

- [ ] **Step 3: Verify the CLI parses**

Run: `cd agent && uv run wiki-agent ask --help`
Expected: help text lists `--no-cache` and `--clear-cache`.

- [ ] **Step 4: Commit**

```bash
git add agent/wiki_agent/cli.py
git commit -m "Add --no-cache/--clear-cache flags to the ask CLI"
```

---

### Task 5: Live rate-limit benchmark

**Files:**
- Create: `agent/wiki_agent/ratelimit_bench.py`
- Create: `agent/tests/test_ratelimit_bench.py`

**Interfaces:**
- Consumes: `config.USER_AGENT/HTTP_TIMEOUT/MAXLAG/WIKI_API`.
- Produces (pure, tested): `ratelimit_bench.Result`, `ratelimit_bench.Summary` dataclasses; `_percentile(values, pct)->float`; `_summarize(name, results, wall)->Summary`; `_pick_best(summaries)->Summary`. Live (untested): `_run_setup(setup)->list[Result]`, `main()`.

- [ ] **Step 1: Write the failing tests** in `agent/tests/test_ratelimit_bench.py`

```python
"""Unit tests for the benchmark's pure scoring helpers — no network."""

from wiki_agent import ratelimit_bench as rb
from wiki_agent.ratelimit_bench import Result, Summary


def _r(ok, status=200, maxlag=False, latency=0.1, throttled=False):
    return Result(ok=ok, status=status, maxlag=maxlag, latency=latency, throttled=throttled)


def test_summarize_counts_rate_and_first_throttle():
    results = [_r(True, latency=0.1),
               _r(False, 429, throttled=True, latency=0.2),
               _r(True, latency=0.3)]
    s = rb._summarize("x", results, wall=2.0)
    assert s.sent == 3
    assert s.successes == 2
    assert s.throttled == 1
    assert s.status_429 == 1
    assert s.req_per_s == 1.0          # 2 successes / 2.0s
    assert s.first_throttle_index == 2


def test_summarize_percentiles_ordered():
    results = [_r(True, latency=l) for l in (0.1, 0.2, 0.3, 0.4)]
    s = rb._summarize("x", results, wall=1.0)
    assert s.p50 <= s.p95


def test_pick_best_prefers_no_throttle_then_throughput():
    fast_throttled = Summary(name="fast", sent=10, successes=10, throttled=3,
        status_429=3, status_503=0, maxlag=0, timeouts=0, p50=0.1, p95=0.2,
        wall=1.0, req_per_s=10.0, first_throttle_index=2)
    clean_slow = Summary(name="clean", sent=5, successes=5, throttled=0,
        status_429=0, status_503=0, maxlag=0, timeouts=0, p50=0.1, p95=0.2,
        wall=1.0, req_per_s=5.0, first_throttle_index=None)
    assert rb._pick_best([fast_throttled, clean_slow]).name == "clean"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd agent && uv run pytest tests/test_ratelimit_bench.py -q`
Expected: FAIL — module/attributes missing.

- [ ] **Step 3: Implement `agent/wiki_agent/ratelimit_bench.py`**

```python
"""Live, opt-in benchmark comparing MediaWiki client setups for throttling.

Run once to choose the best anonymous setup:

    python -m wiki_agent.ratelimit_bench

It hammers the live API (cache bypassed) under four setups and prints a table
plus the recommended config. Bounded and polite: <=500 requests/setup fired in
2-second probe windows, early-stop on sustained throttling, 60s cooldown
between setups. The pure scoring helpers (_summarize/_pick_best/_percentile)
are unit-tested; the live runner is not (no network in the test suite).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from . import config

# Bounded, polite probe parameters.
BUDGET_PER_SETUP = 500
WINDOW_SECONDS = 2.0
WINDOW_PAUSE = 0.5
COOLDOWN_SECONDS = 60.0
EARLY_STOP_CONSECUTIVE = 10
RETRY_STATUS = {429, 503}

# A fixed, varied workload cycled identically across setups.
QUERIES = [
    ("search", "Moon landing"),
    ("get_article", "Apollo 11"),
    ("search", "Theory of relativity"),
    ("get_article", "Albert Einstein"),
    ("search", "Photosynthesis"),
    ("get_article", "DNA"),
    ("search", "French Revolution"),
    ("get_article", "Mount Everest"),
    ("search", "Quantum mechanics"),
    ("get_article", "Pacific Ocean"),
]


@dataclass
class Setup:
    name: str
    concurrency: int
    maxlag: bool


SETUPS = [
    Setup("serial, no maxlag", 1, False),
    Setup("serial, maxlag=5", 1, True),
    Setup("conc=3, no maxlag", 3, False),
    Setup("conc=3, maxlag=5", 3, True),
]


@dataclass
class Result:
    ok: bool
    status: int       # HTTP status, or 0 on transport error
    maxlag: bool      # 200 body that was actually a maxlag rejection
    latency: float    # seconds
    throttled: bool   # 429/503/maxlag


@dataclass
class Summary:
    name: str
    sent: int
    successes: int
    throttled: int
    status_429: int
    status_503: int
    maxlag: int
    timeouts: int
    p50: float
    p95: float
    wall: float
    req_per_s: float
    first_throttle_index: int | None


def _params(kind: str, term: str, maxlag: bool) -> dict:
    base = {"format": "json", "formatversion": 2}
    if maxlag:
        base["maxlag"] = config.MAXLAG
    if kind == "search":
        return {**base, "action": "query", "list": "search",
                "srsearch": term, "srlimit": 5}
    return {**base, "action": "query", "prop": "extracts", "titles": term,
            "exintro": 1, "explaintext": 1, "exchars": 1500, "redirects": 1}


def _one_request(client: httpx.Client, kind: str, term: str, maxlag: bool) -> Result:
    start = time.monotonic()
    try:
        resp = client.get(config.WIKI_API, params=_params(kind, term, maxlag))
    except httpx.HTTPError:
        return Result(False, 0, False, time.monotonic() - start, False)
    latency = time.monotonic() - start
    if resp.status_code in RETRY_STATUS:
        return Result(False, resp.status_code, False, latency, True)
    if resp.status_code != 200:
        return Result(False, resp.status_code, False, latency, False)
    is_maxlag = resp.json().get("error", {}).get("code") == "maxlag"
    return Result(not is_maxlag, 200, is_maxlag, latency, is_maxlag)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarize(name: str, results: list[Result], wall: float) -> Summary:
    lat = [r.latency for r in results]
    successes = sum(1 for r in results if r.ok)
    first = next((i for i, r in enumerate(results, 1) if r.throttled), None)
    return Summary(
        name=name,
        sent=len(results),
        successes=successes,
        throttled=sum(1 for r in results if r.throttled),
        status_429=sum(1 for r in results if r.status == 429),
        status_503=sum(1 for r in results if r.status == 503),
        maxlag=sum(1 for r in results if r.maxlag),
        timeouts=sum(1 for r in results if r.status == 0),
        p50=_percentile(lat, 0.5),
        p95=_percentile(lat, 0.95),
        wall=wall,
        req_per_s=(successes / wall if wall > 0 else 0.0),
        first_throttle_index=first,
    )


def _pick_best(summaries: list[Summary]) -> Summary:
    """Fewest throttle events first, then highest successful throughput."""
    return min(summaries, key=lambda s: (s.throttled, -s.req_per_s))


def _run_setup(setup: Setup) -> list[Result]:
    results: list[Result] = []
    consecutive = 0
    idx = 0
    client = httpx.Client(
        headers={"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip"},
        timeout=config.HTTP_TIMEOUT,
    )
    try:
        pool = ThreadPoolExecutor(max_workers=setup.concurrency) if setup.concurrency > 1 else None
        while len(results) < BUDGET_PER_SETUP and consecutive < EARLY_STOP_CONSECUTIVE:
            window_end = time.monotonic() + WINDOW_SECONDS
            batch: list[Result] = []
            while time.monotonic() < window_end and len(results) + len(batch) < BUDGET_PER_SETUP:
                jobs = []
                for _ in range(setup.concurrency):
                    kind, term = QUERIES[idx % len(QUERIES)]
                    idx += 1
                    jobs.append((kind, term))
                if pool is None:
                    batch.append(_one_request(client, *jobs[0], setup.maxlag))
                else:
                    futures = [pool.submit(_one_request, client, k, t, setup.maxlag)
                               for k, t in jobs]
                    batch.extend(f.result() for f in futures)
            results.extend(batch)
            for r in batch:
                consecutive = consecutive + 1 if r.throttled else 0
                if consecutive >= EARLY_STOP_CONSECUTIVE:
                    break
            if any(r.throttled for r in batch):
                time.sleep(5.0)       # polite backoff after a throttled window
            else:
                time.sleep(WINDOW_PAUSE)
        if pool is not None:
            pool.shutdown(wait=True)
    finally:
        client.close()
    return results


def _print_table(summaries: list[Summary], best: Summary) -> None:
    header = (f"{'setup':<18} {'sent':>5} {'ok':>5} {'thr':>4} {'429':>4} "
              f"{'503':>4} {'lag':>4} {'p50ms':>7} {'p95ms':>7} {'req/s':>7} {'1st-thr':>8}")
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        ft = "-" if s.first_throttle_index is None else str(s.first_throttle_index)
        print(f"{s.name:<18} {s.sent:>5} {s.successes:>5} {s.throttled:>4} "
              f"{s.status_429:>4} {s.status_503:>4} {s.maxlag:>4} "
              f"{s.p50*1000:>7.0f} {s.p95*1000:>7.0f} {s.req_per_s:>7.2f} {ft:>8}")
    print(f"\nRecommended setup: {best.name} "
          f"({best.req_per_s:.2f} successful req/s, {best.throttled} throttle events)")


def main() -> None:
    config.CACHE_ENABLED = False  # always measure the live API
    summaries: list[Summary] = []
    for i, setup in enumerate(SETUPS):
        print(f"Running setup {i+1}/{len(SETUPS)}: {setup.name} ...", flush=True)
        start = time.monotonic()
        results = _run_setup(setup)
        summaries.append(_summarize(setup.name, results, time.monotonic() - start))
        if i < len(SETUPS) - 1:
            print(f"Cooldown {COOLDOWN_SECONDS:.0f}s ...", flush=True)
            time.sleep(COOLDOWN_SECONDS)
    _print_table(summaries, _pick_best(summaries))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd agent && uv run pytest tests/test_ratelimit_bench.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Full suite green**

Run: `cd agent && uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/wiki_agent/ratelimit_bench.py agent/tests/test_ratelimit_bench.py
git commit -m "Add live rate-limit benchmark with pure scoring helpers"
```

---

### Task 6: Run the live experiment + document

**Files:**
- Modify: `CLAUDE.md`
- (No code; produces results to report.)

- [ ] **Step 1: Run the benchmark live**

Run: `cd agent && uv run python -m wiki_agent.ratelimit_bench`
Expected: progress lines, ~5–7 min wall time, then a comparison table and a "Recommended setup" line. Capture the output.

- [ ] **Step 2: Document cache + backoff + bench in `CLAUDE.md`**

Under the `agent/` commands block, add:

```bash
uv run wiki-agent ask "..." --no-cache      # bypass the Wikipedia disk cache
uv run wiki-agent ask "..." --clear-cache   # clear cache, then run
uv run python -m wiki_agent.ratelimit_bench # live rate-limit comparison (opt-in)
```

And add a short note in the file map for `cache.py` and `ratelimit_bench.py`, plus a line under "Models & environment" that the tool uses `maxlag=5`, Retry-After/exponential backoff, and an isolated `agent/.wiki_cache/` (gitignored, no eviction).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document Wikipedia cache, backoff, and rate-limit benchmark"
```

---

## Self-Review

**Spec coverage:**
- Throughput config (UA, gzip, maxlag, serial) → Task 1 (config/UA), gzip already set, serial is inherent. ✓
- Backoff (429/503 + maxlag body, Retry-After/exponential) → Task 3. ✓
- Cache (isolated, simple, clean, no eviction) → Task 2; integration Task 3; CLI Task 4; gitignore Task 2. ✓
- Empirical benchmark (4 setups, 500/setup, 2s windows, 60s cooldown, early-stop, metrics, pick-best) → Task 5; run in Task 6. ✓
- Tests offline → Tasks 2/3/5. ✓
- Docs → Task 6. ✓

**Placeholder scan:** No TBD/TODO; all steps carry real code and exact commands. ✓

**Type consistency:** `Result`/`Summary` fields used identically across `_summarize`, `_pick_best`, `_print_table`, and the tests; `_get`/`_retry_delay`/`_is_maxlag`/`_sleep` signatures match between Task 3 code and tests; `cache._key/get/set/clear` consistent across Tasks 2–4. ✓
