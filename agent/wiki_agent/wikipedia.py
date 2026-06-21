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
from concurrent.futures import ThreadPoolExecutor

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
        "you are unsure of the exact title. To look up several things at once, "
        "pass a `queries` list (with action='search') or a `titles` list (with "
        "action='get_article') in one call — they run in parallel. "
        "Wikipedia has a separate edition per language: set `lang` to the "
        "language code of the country/topic (e.g. hu=Hungarian, is=Icelandic, "
        "et=Estonian, sw=Swahili, hy=Armenian, cy=Welsh, eu=Basque, ka=Georgian, "
        "yo=Yoruba, de, fr, es, ja). Local facts and people are often only on, or "
        "richer on, their native-language edition — query that edition directly."
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
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Several search terms to run in parallel in one call (use instead of 'query').",
            },
            "titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Several exact article titles to fetch in parallel in one call (use instead of 'title').",
            },
            "lang": {
                "type": "string",
                "description": "Wikipedia language edition code (default 'en'). E.g. hu, is, et, sw, hy, cy, eu, ka, yo, de, fr, es, ja.",
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
    """Format a ``generator=search&prop=extracts`` JSON response into readable text.

    Each hit carries a short intro extract, so the agent gets readable content
    straight from the search call (one fewer round-trip than search-then-read).
    Results are ordered by the generator ``index``.
    """
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return f"No Wikipedia articles found for '{query}'."
    pages = sorted(pages, key=lambda p: p.get("index", 0))
    lines = [f"Search results for '{query}':"]
    for i, page in enumerate(pages, start=1):
        title = page.get("title", "(untitled)")
        extract = (page.get("extract") or "").strip()
        lines.append(f"{i}. {title} — {extract}" if extract else f"{i}. {title}")
    return "\n".join(lines)


def _parse_extract(data: dict, title: str, lang: str = config.DEFAULT_LANG) -> str:
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
    url = f"https://{lang}.wikipedia.org/wiki/" + resolved.replace(" ", "_")
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


def _api_url(lang: str) -> str:
    """API endpoint for a given Wikipedia language edition (e.g. 'hu' -> hu.wiki)."""
    return config.WIKI_API_TEMPLATE.format(lang=lang)


def _get(params: dict, client: httpx.Client, lang: str = config.DEFAULT_LANG) -> dict:
    """MediaWiki API GET with disk cache + Retry-After/exponential backoff.

    ``params`` are the semantic query params; transport params (format,
    formatversion, maxlag) are added here and excluded from the cache key. The
    ``lang`` (Wikipedia edition) IS part of the cache key, so the same title on
    different editions never collides.
    """
    cache_params = {"__lang__": lang, **params}
    cached = cache.get(cache_params)
    if cached is not None:
        return cached

    url = _api_url(lang)
    base = {"format": "json", "formatversion": 2, "maxlag": config.MAXLAG}
    for attempt in range(config.MAX_RETRIES + 1):
        response = client.get(url, params={**base, **params})
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
        cache.set(cache_params, data)
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


def search(query: str, limit: int = config.DEFAULT_SEARCH_LIMIT, *,
           lang: str = config.DEFAULT_LANG, client: httpx.Client | None = None) -> str:
    """Full-text search a Wikipedia edition; return a numbered list of titles."""
    owns = client is None
    client = client or _make_client()
    try:
        data = _get(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exchars": config.SEARCH_EXTRACT_CHARS,
            },
            client,
            lang,
        )
        return _parse_search(data, query)
    except httpx.HTTPError as exc:
        return f"Wikipedia search failed: {exc}"
    finally:
        if owns:
            client.close()


def get_article(title: str, chars: int = config.DEFAULT_EXTRACT_CHARS, *,
                lang: str = config.DEFAULT_LANG, client: httpx.Client | None = None) -> str:
    """Fetch the plain-text extract of one Wikipedia article (intro + body).

    Includes article body, not just the lead: many specific facts (populations,
    founding years, dates) live below the intro, so ``exintro`` is intentionally
    omitted and the content is truncated to ``chars`` instead.
    """
    owns = client is None
    client = client or _make_client()
    try:
        data = _get(
            {
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": 1,
                "exchars": chars,
                "redirects": 1,
            },
            client,
            lang,
        )
        return _parse_extract(data, title, lang)
    except httpx.HTTPError as exc:
        return f"Wikipedia fetch failed: {exc}"
    finally:
        if owns:
            client.close()


def _truncate(items: list[str]) -> tuple[list[str], str]:
    """Cap a batch to MAX_BATCH; return (kept, note) with a note when dropping."""
    if len(items) <= config.MAX_BATCH:
        return list(items), ""
    dropped = len(items) - config.MAX_BATCH
    return list(items[: config.MAX_BATCH]), (
        f"\n\n(Note: {dropped} extra item(s) beyond the {config.MAX_BATCH}-lookup "
        "limit were skipped.)"
    )


def search_many(queries: list[str], limit: int = config.DEFAULT_SEARCH_LIMIT, *,
                lang: str = config.DEFAULT_LANG, client: httpx.Client | None = None) -> str:
    """Run several searches in parallel; return labeled, concatenated results."""
    queries, note = _truncate(queries)
    owns = client is None
    client = client or _make_client()
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            results = list(pool.map(lambda q: search(q, limit, lang=lang, client=client), queries))
    finally:
        if owns:
            client.close()
    blocks = [f"=== search: {q!r} ===\n{r}" for q, r in zip(queries, results)]
    return "\n\n".join(blocks) + note


def get_articles(titles: list[str], chars: int = config.DEFAULT_EXTRACT_CHARS, *,
                 lang: str = config.DEFAULT_LANG, client: httpx.Client | None = None) -> str:
    """Fetch several articles in parallel; return labeled, concatenated results."""
    titles, note = _truncate(titles)
    owns = client is None
    client = client or _make_client()
    try:
        with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENCY) as pool:
            results = list(pool.map(lambda t: get_article(t, chars, lang=lang, client=client), titles))
    finally:
        if owns:
            client.close()
    blocks = [f"=== article: {t!r} ===\n{r}" for t, r in zip(titles, results)]
    return "\n\n".join(blocks) + note


def dispatch(tool_input: dict, *, client: httpx.Client | None = None) -> str:
    """Route a tool-call payload to the right action. Returns a readable string."""
    action = tool_input.get("action")
    lang = tool_input.get("lang") or config.DEFAULT_LANG
    if action == "search":
        queries = tool_input.get("queries")
        if queries:
            return search_many(queries, tool_input.get("limit", config.DEFAULT_SEARCH_LIMIT), lang=lang, client=client)
        query = tool_input.get("query")
        if not query:
            return "Error: action='search' requires a 'query' or 'queries'."
        return search(query, tool_input.get("limit", config.DEFAULT_SEARCH_LIMIT), lang=lang, client=client)
    if action == "get_article":
        titles = tool_input.get("titles")
        if titles:
            return get_articles(titles, tool_input.get("chars", config.DEFAULT_EXTRACT_CHARS), lang=lang, client=client)
        title = tool_input.get("title")
        if not title:
            return "Error: action='get_article' requires a 'title' or 'titles'."
        return get_article(title, tool_input.get("chars", config.DEFAULT_EXTRACT_CHARS), lang=lang, client=client)
    return f"Error: unknown action '{action}'. Use 'search' or 'get_article'."
