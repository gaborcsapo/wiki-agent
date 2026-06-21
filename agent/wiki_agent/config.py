"""Single source of truth for tunable constants.

Swapping the model from Haiku to Sonnet is a one-line change here.
"""

# Models. Start on Haiku; switch to "claude-sonnet-4-6" to upgrade.
AGENT_MODEL = "claude-haiku-4-5"

# Claude request limits.
MAX_TOKENS = 2048
DEFAULT_MAX_STEPS = 6

# MediaWiki API.
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia policy requires a descriptive User-Agent with contact info.
USER_AGENT = (
    "WikiAgent/0.1 (https://github.com/gaborxcsapo/anthropic-takehome; "
    "gaborxcsapo@gmail.com) httpx"
)
HTTP_TIMEOUT = 15.0

# Tool defaults.
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_EXTRACT_CHARS = 1500
