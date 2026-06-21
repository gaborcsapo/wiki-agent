"""Unit tests for the Wikipedia tool — pure parsers and routing, no network."""

from wiki_agent import wikipedia


def test_parse_search_formats_numbered_list_and_strips_html():
    data = {
        "query": {
            "search": [
                {"title": "Moon landing", "snippet": 'first <span class="searchmatch">Moon</span> landing'},
                {"title": "Apollo 11", "snippet": "crewed &amp; landed"},
            ]
        }
    }
    out = wikipedia._parse_search(data, "moon landing")
    assert "1. Moon landing — first Moon landing" in out
    assert "2. Apollo 11 — crewed & landed" in out
    assert "<span" not in out  # HTML stripped


def test_parse_search_empty():
    out = wikipedia._parse_search({"query": {"search": []}}, "zzzqqq")
    assert "No Wikipedia articles found" in out


def test_parse_extract_returns_title_url_and_text():
    data = {"query": {"pages": [{"title": "Apollo 11", "extract": "Apollo 11 was a spaceflight."}]}}
    out = wikipedia._parse_extract(data, "Apollo 11")
    assert out.startswith("Apollo 11\n")
    assert "https://en.wikipedia.org/wiki/Apollo_11" in out
    assert "Apollo 11 was a spaceflight." in out


def test_parse_extract_missing_page():
    data = {"query": {"pages": [{"title": "Nonexistent", "missing": True}]}}
    out = wikipedia._parse_extract(data, "Nonexistent")
    assert "No Wikipedia article titled 'Nonexistent' exists" in out
    assert "search" in out  # nudges the model to recover


def test_parse_extract_no_pages():
    out = wikipedia._parse_extract({"query": {"pages": []}}, "X")
    assert "No article found" in out


def test_dispatch_requires_query_for_search():
    assert wikipedia.dispatch({"action": "search"}).startswith("Error")


def test_dispatch_requires_title_for_get_article():
    assert wikipedia.dispatch({"action": "get_article"}).startswith("Error")


def test_dispatch_unknown_action():
    out = wikipedia.dispatch({"action": "delete_everything"})
    assert "unknown action" in out


def test_dispatch_routes_search(monkeypatch):
    captured = {}

    def fake_search(query, limit, *, client=None):
        captured["query"] = query
        captured["limit"] = limit
        return "results"

    monkeypatch.setattr(wikipedia, "search", fake_search)
    out = wikipedia.dispatch({"action": "search", "query": "cats", "limit": 3})
    assert out == "results"
    assert captured == {"query": "cats", "limit": 3}


def test_dispatch_routes_get_article(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_article", lambda title, chars, *, client=None: f"got:{title}:{chars}")
    out = wikipedia.dispatch({"action": "get_article", "title": "Cat", "chars": 200})
    assert out == "got:Cat:200"
