"""Benchmark registry. One @task per benchmark.

Adding a new benchmark = drop a JSONL dataset in datasets/ and add a @task here
(optionally with benchmark-specific scorers).
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset

# Absolute imports (not relative): Inspect loads this file by path, not as
# `wiki_eval.tasks`, so relative imports have no package to resolve against.
from wiki_eval.scorers import (
    abstention_judge,
    correctness_judge,
    retrieval_grounding,
    used_wikipedia_tool,
)
from wiki_eval.solver import wiki_agent_solver

_DATASETS = Path(__file__).parent / "datasets"


@task
def factual_qa():
    """Factual question answering over Wikipedia (10 hand-written examples)."""
    return Task(
        dataset=json_dataset(str(_DATASETS / "factual_qa.jsonl")),
        solver=wiki_agent_solver(),
        scorer=[correctness_judge(), used_wikipedia_tool()],
    )


def _frames_record_to_sample(record: dict) -> Sample:
    """Map a FRAMES jsonl row to a Sample, carrying gold pages into metadata.

    `reference_pages` is read only by the retrieval_grounding scorer, which is
    why grounding stays FRAMES-only: no other dataset sets this key.
    """
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={"reference_pages": record.get("reference_pages", [])},
    )


@task
def frames():
    """FRAMES multi-hop Wikipedia QA, with a retrieval-grounding (recall) signal.

    FRAMES questions need 2-15 article reads, so the agent gets a larger step
    budget than the small benchmarks' default.
    """
    return Task(
        dataset=json_dataset(
            str(_DATASETS / "frames.jsonl"),
            sample_fields=_frames_record_to_sample,
        ),
        solver=wiki_agent_solver(max_steps=20),
        scorer=[correctness_judge(), used_wikipedia_tool(), retrieval_grounding()],
    )


def _abstention_record_to_sample(record: dict) -> Sample:
    """Map an abstention jsonl row to a Sample, carrying the abstention labels.

    `should_abstain` and `category` are read by the abstention_judge scorer (and
    its precision/recall/F1 metrics); no other dataset sets these keys.
    """
    return Sample(
        input=record["input"],
        target=record["target"],
        metadata={
            "should_abstain": record["should_abstain"],
            "category": record["category"],
        },
    )


@task
def wiki_agent_abstention():
    """Abstention: should the agent decline/flag instead of hallucinating?

    30 abstention-positive questions (false premise, unknowable, stale/real-time,
    underspecified, subjective, garbled, out-of-scope) + 6 answerable controls.
    Graded by a binary abstention judge against each row's `should_abstain`,
    reported as abstention precision/recall/F1 + accuracy. `used_wikipedia_tool`
    rides along as a diagnostic (it is NOT a pass/fail signal here).
    """
    return Task(
        dataset=json_dataset(
            str(_DATASETS / "abstention.jsonl"),
            sample_fields=_abstention_record_to_sample,
        ),
        solver=wiki_agent_solver(),
        scorer=[abstention_judge(), used_wikipedia_tool()],
    )
