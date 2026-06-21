"""Unit test for the multilingual analyzer's pure aggregator — no logs/network."""

from wiki_eval.analyze_multilingual import _summarize


def _rec(correct, category, language):
    return {"correct": correct, "category": category, "language": language}


def test_summarize_overall_and_grouped():
    records = [
        _rec(True, "cross_lingual_fact", "hu"),
        _rec(False, "cross_lingual_fact", "hu"),
        _rec(True, "richer_native_page", "is"),
    ]
    s = _summarize(records)
    assert s["n"] == 3
    assert abs(s["accuracy"] - 2 / 3) < 1e-9
    assert s["by_category"]["cross_lingual_fact"] == 0.5
    assert s["by_category"]["richer_native_page"] == 1.0
    assert s["by_language"]["hu"] == 0.5
    assert s["by_language"]["is"] == 1.0


def test_summarize_empty():
    s = _summarize([])
    assert s["n"] == 0 and s["accuracy"] == 0.0
