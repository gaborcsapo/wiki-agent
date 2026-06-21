"""Unit tests for the custom trajectory scorer (no API calls)."""

import asyncio
from types import SimpleNamespace

from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, Target

from wiki_eval.scorers import correctness_judge, used_wikipedia_tool


def _state(steps):
    return SimpleNamespace(metadata={"trajectory": {"steps": steps}})


# --- correctness_judge (LLM-as-judge, offline via mockllm) ----------------


def _judge_state(question: str, answer: str):
    return SimpleNamespace(
        input_text=question,
        output=ModelOutput.from_content("agent", answer),
        metadata={},
    )


def _mock_grader(grade: str):
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", grade)],
    )


def test_correctness_judge_grades_correct():
    state = _judge_state("Who was first on the Moon?", "Neil Armstrong.")
    score = asyncio.run(
        correctness_judge(model=_mock_grader("GRADE: C"))(state, Target("Neil Armstrong"))
    )
    assert score.value == CORRECT


def test_correctness_judge_grades_incorrect():
    state = _judge_state("Who was first on the Moon?", "Buzz Aldrin.")
    score = asyncio.run(
        correctness_judge(model=_mock_grader("GRADE: I"))(state, Target("Neil Armstrong"))
    )
    assert score.value == INCORRECT


def test_used_tool_true_when_tool_called():
    score = asyncio.run(used_wikipedia_tool()(_state([{"kind": "tool_call"}]), None))
    assert score.value == CORRECT


def test_used_tool_false_without_tool_call():
    score = asyncio.run(used_wikipedia_tool()(_state([{"kind": "assistant_text"}]), None))
    assert score.value == INCORRECT


def test_used_tool_false_when_no_trajectory():
    score = asyncio.run(used_wikipedia_tool()(SimpleNamespace(metadata={}), None))
    assert score.value == INCORRECT
