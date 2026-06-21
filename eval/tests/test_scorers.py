"""Unit tests for the custom trajectory scorer (no API calls)."""

import asyncio
from types import SimpleNamespace

from inspect_ai.scorer import CORRECT, INCORRECT

from wiki_eval.scorers import used_wikipedia_tool


def _state(steps):
    return SimpleNamespace(metadata={"trajectory": {"steps": steps}})


def test_used_tool_true_when_tool_called():
    score = asyncio.run(used_wikipedia_tool()(_state([{"kind": "tool_call"}]), None))
    assert score.value == CORRECT


def test_used_tool_false_without_tool_call():
    score = asyncio.run(used_wikipedia_tool()(_state([{"kind": "assistant_text"}]), None))
    assert score.value == INCORRECT


def test_used_tool_false_when_no_trajectory():
    score = asyncio.run(used_wikipedia_tool()(SimpleNamespace(metadata={}), None))
    assert score.value == INCORRECT
