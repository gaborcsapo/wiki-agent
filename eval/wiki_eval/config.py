"""Eval configuration — single source of truth for model choices.

Switch either line to "anthropic/claude-sonnet-4-6" (judge) or
"claude-sonnet-4-6" (agent) to upgrade.
"""

# The grader/judge model, as an Inspect provider string.
JUDGE_MODEL = "anthropic/claude-haiku-4-5"

# The model the agent-under-test uses. None => the agent's own default (Haiku).
# This is the agent's native model id (e.g. "claude-haiku-4-5"), not a provider string.
AGENT_MODEL = None
