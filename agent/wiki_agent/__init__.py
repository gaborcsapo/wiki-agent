"""wiki_agent — a minimal Wikipedia question-answering agent.

The public surface is intentionally tiny: ``run`` and ``AgentResult``. This is
the single, clean boundary the evaluation suite depends on.
"""

from .agent import AgentResult, run

__all__ = ["run", "AgentResult"]
