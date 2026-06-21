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
