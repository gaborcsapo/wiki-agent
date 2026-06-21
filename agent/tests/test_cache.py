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
