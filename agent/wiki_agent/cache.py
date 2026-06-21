"""A tiny, isolated disk cache for raw MediaWiki API JSON responses.

Keyed by the semantic request params so identical lookups across benchmark
re-runs are served from disk instead of re-fetched. Deliberately simple: no
TTL, no eviction, no size management. Never raises into callers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

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
    # Write to a unique temp file then atomically rename, so a crash or two
    # concurrent writers of the same key can never leave a truncated file (which
    # would be a permanent cache poison — there is no eviction).
    try:
        fd, tmp = tempfile.mkstemp(dir=config.CACHE_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001 — caching is best-effort, never raise
            try:
                os.unlink(tmp)
            except OSError:
                pass
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
