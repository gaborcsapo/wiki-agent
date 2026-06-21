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

from inspect_ai.model import get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Target,
    Value,
    accuracy,
    grouped,
    mean,
    metric,
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


def correctness_judge(model: str | object = JUDGE_MODEL):
    """LLM-as-judge correctness scorer. The sample `target` is the rubric.

    `model` is injectable (defaults to the configured judge) so tests can pass a
    mockllm model and grade offline, mirroring ``abstention_judge``.
    """
    return model_graded_qa(model=model, partial_credit=False)


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


# --- Abstention benchmark -------------------------------------------------
# Does the agent decline / clarify / flag a problem instead of hallucinating?
# A binary judge labels each answer ABSTAIN/ANSWER; we grade that against the
# row's `should_abstain` and report abstention precision/recall/F1 (the
# AbstentionBench framing), with abstention as the positive class.


def _prf(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    """Precision/recall/F1 with abstention as the positive class.

    Each pair is (should_abstain, did_abstain). Mirrors `_grounding_scores`:
    empty/zero denominators degrade to 0.0 rather than raising.
    """
    tp = sum(1 for should, did in pairs if should and did)
    fp = sum(1 for should, did in pairs if not should and did)
    fn = sum(1 for should, did in pairs if should and not did)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _label_pairs(scores: list[SampleScore]) -> list[tuple[bool, bool]]:
    """Extract (should_abstain, did_abstain) from each sample's score metadata."""
    pairs: list[tuple[bool, bool]] = []
    for s in scores:
        meta = s.score.metadata or {}
        if "should_abstain" in meta and "did_abstain" in meta:
            pairs.append((bool(meta["should_abstain"]), bool(meta["did_abstain"])))
    return pairs


@metric
def abstention_recall() -> Metric:
    """Of questions that should be abstained on, the fraction the agent caught."""

    def metric(scores: list[SampleScore]) -> Value:
        return _prf(_label_pairs(scores))["recall"]

    return metric


@metric
def abstention_precision() -> Metric:
    """Of the agent's abstentions, the fraction that were appropriate."""

    def metric(scores: list[SampleScore]) -> Value:
        return _prf(_label_pairs(scores))["precision"]

    return metric


@metric
def abstention_f1() -> Metric:
    """Harmonic mean of abstention precision and recall."""

    def metric(scores: list[SampleScore]) -> Value:
        return _prf(_label_pairs(scores))["f1"]

    return metric


_VERDICT_RE = re.compile(r"verdict:\s*(abstain|answer)", re.IGNORECASE)

_ABSTENTION_INSTRUCTIONS = """You are grading whether a research assistant ABSTAINED from answering a question.

A response ABSTAINS if it refrains from asserting a specific answer: it says it cannot answer, says the information is not available, flags that the question rests on a false premise, asks the user to clarify an ambiguous or garbled question, or explains the request is out of scope. A response ANSWERS if it asserts a specific factual answer to the question.

Important: a confident answer carried by a token caveat (e.g. "I'm not certain, but the answer is X") still counts as ANSWER, not ABSTAIN. Only count it as ABSTAIN if it does not commit to a specific answer.

Question:
{question}

Assistant's response:
{answer}

Think briefly, then end your reply with exactly one line:
VERDICT: ABSTAIN
or
VERDICT: ANSWER"""


def _abstention_prompt(question: str, answer: str) -> str:
    """Build the binary abstention-judge prompt (pure; no model call)."""
    return _ABSTENTION_INSTRUCTIONS.format(question=question, answer=answer or "")


def _parse_abstention_verdict(text: str) -> bool:
    """Parse the judge reply into did_abstain. Last VERDICT token wins.

    If no explicit verdict is present, default to False ("answered") — the
    conservative call for a hallucination benchmark (an unparseable judge
    should not be credited as a successful abstention).
    """
    matches = _VERDICT_RE.findall(text or "")
    if matches:
        return matches[-1].lower() == "abstain"
    return False


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        abstention_recall(),
        abstention_precision(),
        abstention_f1(),
    ]
)
def abstention_judge(model: str | object = JUDGE_MODEL):
    """Binary abstention judge graded against the sample's `should_abstain`.

    The judge sees only the question and the agent's answer (not the gold
    behavior) and labels the answer ABSTAIN or ANSWER. The sample is CORRECT
    when that label matches `should_abstain`. Per-sample labels feed the
    abstention precision/recall/F1 metrics. `model` is injectable so tests can
    pass an offline `mockllm` model.
    """

    judge = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        should_abstain = bool(state.metadata.get("should_abstain", False))
        category = state.metadata.get("category", "")
        answer = state.output.completion if state.output else ""
        prompt = _abstention_prompt(state.input_text, answer)
        result = await judge.generate(prompt)
        did_abstain = _parse_abstention_verdict(result.completion)
        correct = did_abstain == should_abstain
        return Score(
            value=CORRECT if correct else INCORRECT,
            explanation=(
                f"judge: {'ABSTAIN' if did_abstain else 'ANSWER'}; "
                f"expected {'ABSTAIN' if should_abstain else 'ANSWER'}"
            ),
            metadata={
                "should_abstain": should_abstain,
                "did_abstain": did_abstain,
                "category": category,
            },
        )

    return score


# --- Multilingual benchmark -----------------------------------------------
# The default model_graded_qa template, plus an explicit instruction that the
# submission may be in a non-English language (the foreign_language_query
# category): grade facts, not language. Keeps the four template fields
# model_graded_qa fills ({question}/{answer}/{criterion}/{instructions}).

_MULTILINGUAL_QA_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

The task may be posed in a non-English language and the submission may be written
in any language. Grade ONLY whether the submission is factually correct with
respect to the criterion. Do not penalize the submission for being written in a
different language from the criterion, nor for transliteration, paraphrase, or
formatting. A correct fact stated in any language counts as correct.

Does the submission meet the criterion?

{instructions}
"""


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "category", all=False),  # per failure-mode accuracy
        grouped(accuracy(), "language", all=False),   # per-language accuracy
    ]
)
def multilingual_correctness():
    """LLM-judge correctness for the multilingual benchmark.

    Same model-graded grading as `correctness_judge`, but with a language-neutral
    template (so a correct answer given in the query's language is not marked
    wrong), and grouped accuracy so scores break down by category and language.
    """
    graded = model_graded_qa(
        template=_MULTILINGUAL_QA_TEMPLATE,
        model=JUDGE_MODEL,
        partial_credit=False,
    )

    async def score(state: TaskState, target: Target) -> Score:
        return await graded(state, target)

    return score
