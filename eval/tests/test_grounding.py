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
