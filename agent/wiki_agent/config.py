"""Single source of truth for tunable constants.

Swapping the model from Haiku to Sonnet is a one-line change here.
"""

from pathlib import Path

import httpx

# Models. Start on Haiku; switch to "claude-sonnet-4-6" to upgrade.
AGENT_MODEL = "claude-sonnet-4-6"

# Extended thinking (Sonnet 4.6+ only — Haiku does NOT support it; leave None there).
# THINKING = {"type": "adaptive"} enables adaptive thinking; EFFORT in
# {"low","medium","high","max"} tunes its depth (the modern "thinking budget").
THINKING: dict | None = None
EFFORT: str | None = None

# Claude request limits.
MAX_TOKENS = 2048
DEFAULT_MAX_STEPS = 6

# MediaWiki API.
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia policy requires a descriptive User-Agent with contact info and the
# underlying library/version. A compliant UA grants the 200 req/min tier
# (vs 10 req/min for a generic/empty one).
USER_AGENT = (
    "WikiAgent/0.1 (https://github.com/gaborxcsapo/anthropic-takehome; "
    f"gaborxcsapo@gmail.com) python-httpx/{httpx.__version__}"
)
HTTP_TIMEOUT = 15.0

# MediaWiki etiquette: shed load under DB replica lag on non-interactive traffic.
MAXLAG = 5

# Backoff: Wikimedia asks clients to honor Retry-After, else wait >=5s then
# back off exponentially. Applied in wikipedia._get.
MAX_RETRIES = 4
BACKOFF_BASE = 1.0       # seconds (exponential base)
BACKOFF_CAP = 30.0       # seconds (max single wait)
MIN_RETRY_WAIT = 5.0     # seconds (floor when no Retry-After header)

# Disk cache for raw API JSON. Isolated dir, no eviction/TTL (simplicity over
# size). agent/.wiki_cache (parent.parent of this file = the agent/ root).
CACHE_ENABLED = True
CACHE_DIR = Path(__file__).resolve().parent.parent / ".wiki_cache"

# Tool defaults.
DEFAULT_SEARCH_LIMIT = 5
# Larger extract: FRAMES-style facts often sit in the article body, not the lead.
DEFAULT_EXTRACT_CHARS = 4000
