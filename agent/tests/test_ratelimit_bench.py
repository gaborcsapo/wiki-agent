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
