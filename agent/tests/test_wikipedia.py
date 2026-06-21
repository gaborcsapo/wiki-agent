"""Unit tests for the Wikipedia tool — pure parsers and routing, no network."""

import httpx
import pytest

from wiki_agent import config, wikipedia


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

    def fake_search(query, limit, *, lang=wikipedia.config.DEFAULT_LANG, client=None):
        captured["query"] = query
        captured["limit"] = limit
        return "results"

    monkeypatch.setattr(wikipedia, "search", fake_search)
    out = wikipedia.dispatch({"action": "search", "query": "cats", "limit": 3})
    assert out == "results"
    assert captured == {"query": "cats", "limit": 3}


def test_dispatch_routes_get_article(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_article", lambda title, chars, *, lang=config.DEFAULT_LANG, client=None: f"got:{title}:{chars}")
    out = wikipedia.dispatch({"action": "get_article", "title": "Cat", "chars": 200})
    assert out == "got:Cat:200"


# ---------------------------------------------------------------------------
# Backoff + cache in _get (fakes only, no network / no real sleeps)
# ---------------------------------------------------------------------------


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
    assert second == {"query": {"v": 1}}  # served from cache, no pop from empty


# ---------------------------------------------------------------------------
# Parallel multi-query batch lookups
# ---------------------------------------------------------------------------


def test_truncate_caps_to_max_batch(monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH", 2)
    items, note = wikipedia._truncate(["a", "b", "c", "d"])
    assert items == ["a", "b"]
    assert "2 extra" in note


def test_truncate_no_note_within_limit():
    items, note = wikipedia._truncate(["a", "b"])
    assert items == ["a", "b"] and note == ""


def test_search_many_fans_out_in_order_with_shared_client(monkeypatch):
    calls = []

    def fake_search(q, limit, *, lang=config.DEFAULT_LANG, client=None):
        calls.append((q, limit, client))
        return f"R:{q}"

    monkeypatch.setattr(wikipedia, "search", fake_search)
    out = wikipedia.search_many(["a", "b", "c"], 5, client="shared")
    assert "=== search: 'a' ===\nR:a" in out
    assert out.index("R:a") < out.index("R:b") < out.index("R:c")
    assert [c[0] for c in calls] == ["a", "b", "c"]
    assert all(c[1] == 5 and c[2] == "shared" for c in calls)


def test_get_articles_fans_out(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_article", lambda t, chars, *, lang=config.DEFAULT_LANG, client=None: f"A:{t}:{chars}")
    out = wikipedia.get_articles(["X", "Y"], 100, client="shared")
    assert "=== article: 'X' ===\nA:X:100" in out
    assert "A:Y:100" in out


def test_search_many_truncates(monkeypatch):
    monkeypatch.setattr(config, "MAX_BATCH", 2)
    monkeypatch.setattr(wikipedia, "search", lambda q, limit, *, lang=config.DEFAULT_LANG, client=None: f"R:{q}")
    out = wikipedia.search_many(["a", "b", "c", "d"], client="s")
    assert "R:a" in out and "R:b" in out and "R:c" not in out
    assert "2 extra" in out


def test_dispatch_routes_search_many(monkeypatch):
    monkeypatch.setattr(wikipedia, "search_many", lambda qs, limit, *, lang=config.DEFAULT_LANG, client=None: f"many:{qs}:{limit}")
    out = wikipedia.dispatch({"action": "search", "queries": ["a", "b"], "limit": 3})
    assert out == "many:['a', 'b']:3"


def test_dispatch_routes_get_articles(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_articles", lambda ts, chars, *, lang=config.DEFAULT_LANG, client=None: f"arts:{ts}")
    out = wikipedia.dispatch({"action": "get_article", "titles": ["X", "Y"]})
    assert out == "arts:['X', 'Y']"


def test_dispatch_empty_queries_falls_back_to_single(monkeypatch):
    monkeypatch.setattr(wikipedia, "search", lambda q, limit, *, lang=config.DEFAULT_LANG, client=None: f"one:{q}")
    out = wikipedia.dispatch({"action": "search", "queries": [], "query": "z"})
    assert out == "one:z"


def test_schema_advertises_lists():
    props = wikipedia.TOOL_SCHEMA["input_schema"]["properties"]
    assert props["queries"]["type"] == "array"
    assert props["titles"]["type"] == "array"


# ---------------------------------------------------------------------------
# Multilingual: per-language Wikipedia editions
# ---------------------------------------------------------------------------


def test_api_url_builds_per_language_host():
    assert wikipedia._api_url("hu") == "https://hu.wikipedia.org/w/api.php"
    assert wikipedia._api_url("en") == "https://en.wikipedia.org/w/api.php"


def test_schema_advertises_lang():
    assert wikipedia.TOOL_SCHEMA["input_schema"]["properties"]["lang"]["type"] == "string"


def test_get_requests_per_language_host(monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    seen = {}

    class CapClient:
        def get(self, url, params=None):
            seen["url"] = url
            return FakeResponse(200, {"query": {"ok": True}})

    wikipedia._get({"action": "query"}, CapClient(), "hu")
    assert seen["url"] == "https://hu.wikipedia.org/w/api.php"


def test_get_lang_does_not_collide_in_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_ENABLED", True)
    params = {"action": "query", "titles": "Budapest"}
    en = wikipedia._get(params, FakeClient([FakeResponse(200, {"v": "en"})]), "en")
    hu = wikipedia._get(params, FakeClient([FakeResponse(200, {"v": "hu"})]), "hu")
    assert en == {"v": "en"} and hu == {"v": "hu"}  # distinct cache entries


def test_search_passes_lang_to_get(monkeypatch):
    seen = {}

    def fake_get(params, client, lang=config.DEFAULT_LANG):
        seen["lang"] = lang
        return {"query": {"search": []}}

    monkeypatch.setattr(wikipedia, "_get", fake_get)
    wikipedia.search("x", lang="et", client="c")
    assert seen["lang"] == "et"


def test_dispatch_threads_lang(monkeypatch):
    seen = {}
    monkeypatch.setattr(wikipedia, "get_article", lambda title, chars, *, lang=config.DEFAULT_LANG, client=None: seen.setdefault("lang", lang) or "ok")
    wikipedia.dispatch({"action": "get_article", "title": "Reykjavík", "lang": "is"})
    assert seen["lang"] == "is"


def test_dispatch_defaults_lang_to_en(monkeypatch):
    seen = {}
    monkeypatch.setattr(wikipedia, "search", lambda q, limit, *, lang=config.DEFAULT_LANG, client=None: seen.setdefault("lang", lang) or "ok")
    wikipedia.dispatch({"action": "search", "query": "x"})
    assert seen["lang"] == "en"
