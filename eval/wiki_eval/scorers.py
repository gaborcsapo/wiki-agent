"""Scorers for the Wikipedia agent.

Two kinds, illustrating the two grading styles from Anthropic's eval guidance:

* `correctness_judge` — LLM-as-judge (model-based grading) for the open-ended
  correctness of the answer. This is the primary metric.
* `used_wikipedia_tool` — a code-based / trajectory scorer that checks the agent
  actually used its tool rather than answering from memory.

To add a benchmark-specific metric, write another `@scorer` here (e.g.
groundedness of the answer against the fetched extract) and add it to the task's
scorer list.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    mean,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .config import JUDGE_MODEL

_WIKI_PATH_RE = re.compile(r"^/wiki/(.+)$")
_WIKI_URL_RE = re.compile(r"https?://en\.wikipedia\.org/wiki/\S+")


def _normalize_wiki_url(url: str) -> str | None:
    """Canonical slug for an English-Wikipedia article URL, else None.

    Lowercased, spaces not underscores, percent-decoded, query/fragment dropped.
    """
    parts = urlsplit(url.strip())
    match = _WIKI_PATH_RE.match(parts.path)
    if not match:
        return None
    slug = unquote(match.group(1)).replace("_", " ").strip().casefold()
    return slug or None


def _fetched_pages(steps: list[dict]) -> set[str]:
    """Normalized slugs the agent actually read via get_article.

    Only get_article results carry a canonical `.../wiki/<slug>` URL line; search
    listings and error messages do not, so they contribute nothing.
    """
    pages: set[str] = set()
    for step in steps:
        if step.get("kind") != "tool_result":
            continue
        for raw in _WIKI_URL_RE.findall(step.get("content") or ""):
            slug = _normalize_wiki_url(raw.rstrip(".,);"))
            if slug:
                pages.add(slug)
    return pages


def _grounding_scores(gold: set[str], read: set[str]) -> dict[str, float]:
    """Recall (headline), precision, and F1 of read pages vs. gold pages."""
    if not gold:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    hits = len(gold & read)
    recall = hits / len(gold)
    precision = hits / len(read) if read else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def correctness_judge():
    """LLM-as-judge correctness scorer. The sample `target` is the rubric."""
    return model_graded_qa(model=JUDGE_MODEL, partial_credit=False)


@scorer(metrics=[accuracy(), stderr()])
def used_wikipedia_tool():
    """Code-based scorer: did the agent actually call the wikipedia tool?

    Reads the trajectory the solver attached to `state.metadata`.
    """

    async def score(state: TaskState, target: Target) -> Score:
        steps = state.metadata.get("trajectory", {}).get("steps", [])
        used = any(step.get("kind") == "tool_call" for step in steps)
        return Score(
            value=CORRECT if used else INCORRECT,
            explanation="agent called the wikipedia tool" if used else "no tool call recorded",
        )

    return score


@scorer(metrics={"recall": [mean(), stderr()], "precision": [mean()], "f1": [mean()]})
def retrieval_grounding():
    """FRAMES-only: overlap between pages the agent read and gold reference pages.

    Recall is the headline (did the agent find the needed evidence?); precision
    and F1 ride along as diagnostics. Gold pages come from the sample metadata
    that only the FRAMES loader populates, so this scorer is inert elsewhere.
    """

    async def score(state: TaskState, target: Target) -> Score:
        gold = {
            slug
            for url in state.metadata.get("reference_pages", [])
            if (slug := _normalize_wiki_url(url))
        }
        steps = state.metadata.get("trajectory", {}).get("steps", [])
        read = _fetched_pages(steps)
        values = _grounding_scores(gold, read)
        hits = len(gold & read)
        return Score(
            value=values,
            explanation=f"{hits}/{len(gold)} gold pages read; {len(read)} read total",
            metadata={"n_gold": len(gold), "n_read": len(read), "n_hit": hits},
        )

    return score
