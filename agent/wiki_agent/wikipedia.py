"""The agent's one and only tool: a Wikipedia lookup over the MediaWiki API.

A single tool named ``wikipedia`` exposes two actions:

* ``search``      — full-text search, returns a numbered list of matching titles.
* ``get_article`` — fetch the plain-text intro extract of one article.

Design notes:
* Pure formatting logic (``_parse_search``/``_parse_extract``) is separated from
  HTTP I/O (``_get``) so the parsing can be unit-tested without a network.
* Errors (missing pages, bad input, network failures) are returned as readable
  strings — never raised — so the model can recover gracefully.
"""

from __future__ import annotations

import html
import re
import time

import httpx

from . import cache, config

# ---------------------------------------------------------------------------
# Tool schema (passed to the Claude API)
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "name": "wikipedia",
    "description": (
        "Look things up on English Wikipedia. Use action='search' to find "
        "relevant article titles from a query, then action='get_article' to "
        "read an article's introduction by its exact title. Search first when "
        "you are unsure of the exact title."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "get_article"],
                "description": "Which operation to perform.",
            },
            "query": {
                "type": "string",
                "description": "Search terms. Required when action='search'.",
            },
            "title": {
                "type": "string",
                "description": "Exact article title. Required when action='get_article'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max search results (default 5).",
            },
            "chars": {
                "type": "integer",
                "description": "Max characters of the article extract (default 1500).",
            },
        },
        "required": ["action"],
    },
}

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags (e.g. search-match highlight spans) and unescape entities."""
    return html.unescape(_TAG_RE.sub("", text)).strip()


# ---------------------------------------------------------------------------
# Pure parsers (no network) — unit-tested directly
# ---------------------------------------------------------------------------


def _parse_search(data: dict, query: str) -> str:
    """Format an ``action=query&list=search`` JSON response into readable text."""
    results = data.get("query", {}).get("search", [])
    if not results:
        return f"No Wikipedia articles found for '{query}'."
    lines = [f"Search results for '{query}':"]
    for i, hit in enumerate(results, start=1):
        title = hit.get("title", "(untitled)")
        snippet = _strip_html(hit.get("snippet", ""))
        lines.append(f"{i}. {title} — {snippet}" if snippet else f"{i}. {title}")
    return "\n".join(lines)


def _parse_extract(data: dict, title: str) -> str:
    """Format an ``action=query&prop=extracts`` JSON response into readable text."""
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return f"No article found for '{title}'."
    page = pages[0]
    if page.get("missing"):
        return (
            f"No Wikipedia article titled '{title}' exists. "
            "Try action='search' to find the correct title."
        )
    extract = (page.get("extract") or "").strip()
    resolved = page.get("title", title)
    if not extract:
        return f"The article '{resolved}' has no readable extract."
    url = "https://en.wikipedia.org/wiki/" + resolved.replace(" ", "_")
    return f"{resolved}\n{url}\n\n{extract}"


# ---------------------------------------------------------------------------
# HTTP I/O
# ---------------------------------------------------------------------------


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


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip"},
        timeout=config.HTTP_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------


def search(query: str, limit: int = config.DEFAULT_SEARCH_LIMIT, *, client: httpx.Client | None = None) -> str:
    """Full-text search Wikipedia; return a numbered list of titles + snippets."""
    owns = client is None
    client = client or _make_client()
    try:
        data = _get(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": limit},
            client,
        )
        return _parse_search(data, query)
    except httpx.HTTPError as exc:
        return f"Wikipedia search failed: {exc}"
    finally:
        if owns:
            client.close()


def get_article(title: str, chars: int = config.DEFAULT_EXTRACT_CHARS, *, client: httpx.Client | None = None) -> str:
    """Fetch the plain-text intro extract of one Wikipedia article."""
    owns = client is None
    client = client or _make_client()
    try:
        data = _get(
            {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "exintro": 1,
                "explaintext": 1,
                "exchars": chars,
                "redirects": 1,
            },
            client,
        )
        return _parse_extract(data, title)
    except httpx.HTTPError as exc:
        return f"Wikipedia fetch failed: {exc}"
    finally:
        if owns:
            client.close()


def dispatch(tool_input: dict, *, client: httpx.Client | None = None) -> str:
    """Route a tool-call payload to the right action. Returns a readable string."""
    action = tool_input.get("action")
    if action == "search":
        query = tool_input.get("query")
        if not query:
            return "Error: action='search' requires a 'query'."
        return search(query, tool_input.get("limit", config.DEFAULT_SEARCH_LIMIT), client=client)
    if action == "get_article":
        title = tool_input.get("title")
        if not title:
            return "Error: action='get_article' requires a 'title'."
        return get_article(title, tool_input.get("chars", config.DEFAULT_EXTRACT_CHARS), client=client)
    return f"Error: unknown action '{action}'. Use 'search' or 'get_article'."
