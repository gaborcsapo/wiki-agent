"""Benchmark registry. One @task per benchmark.

Adding a new benchmark = drop a JSONL dataset in datasets/ and add a @task here
(optionally with benchmark-specific scorers).
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset

# Absolute imports (not relative): Inspect loads this file by path, not as
# `wiki_eval.tasks`, so relative imports have no package to resolve against.
from wiki_eval.scorers import correctness_judge, used_wikipedia_tool
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
