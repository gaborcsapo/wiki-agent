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
    try:
        body = resp.json()
    except ValueError:
        body = {}
    is_maxlag = body.get("error", {}).get("code") == "maxlag"
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
    pool = ThreadPoolExecutor(max_workers=setup.concurrency) if setup.concurrency > 1 else None
    try:
        stop = False
        while len(results) < BUDGET_PER_SETUP and not stop:
            window_end = time.monotonic() + WINDOW_SECONDS
            batch: list[Result] = []
            while time.monotonic() < window_end and not stop:
                # Never fire more than the remaining budget, so a concurrent chunk
                # can't push the total past BUDGET_PER_SETUP.
                room = BUDGET_PER_SETUP - len(results) - len(batch)
                if room <= 0:
                    break
                jobs = []
                for _ in range(min(setup.concurrency, room)):
                    kind, term = QUERIES[idx % len(QUERIES)]
                    idx += 1
                    jobs.append((kind, term))
                if pool is None:
                    chunk = [_one_request(client, *jobs[0], setup.maxlag)]
                else:
                    futures = [pool.submit(_one_request, client, k, t, setup.maxlag)
                               for k, t in jobs]
                    chunk = [f.result() for f in futures]
                # Update the consecutive-throttle counter per request and stop the
                # moment the threshold is hit — don't keep firing for the rest of
                # the window once sustained throttling is detected.
                for r in chunk:
                    batch.append(r)
                    consecutive = consecutive + 1 if r.throttled else 0
                    if consecutive >= EARLY_STOP_CONSECUTIVE:
                        stop = True
                        break
            results.extend(batch)
            if stop:
                break
            if any(r.throttled for r in batch):
                time.sleep(5.0)       # polite backoff after a throttled window
            else:
                time.sleep(WINDOW_PAUSE)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
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
