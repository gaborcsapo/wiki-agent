"""Unit test for the agent-wrapping solver, using a fake agent (no API calls)."""

import asyncio
from types import SimpleNamespace

import wiki_agent
from wiki_agent.trajectory import (
    ASSISTANT_TEXT,
    FINAL_ANSWER,
    TOOL_CALL,
    Step,
    Trajectory,
)

from wiki_eval.solver import _trajectory_usage, wiki_agent_solver


def test_trajectory_usage_sums_step_tokens():
    """Pure helper: total the agent's per-step input/output tokens."""
    steps = [
        {"kind": ASSISTANT_TEXT, "input_tokens": 100, "output_tokens": 20},
        {"kind": TOOL_CALL},  # tool steps carry no token counts
        {"kind": ASSISTANT_TEXT, "input_tokens": None, "output_tokens": None},
        {"kind": FINAL_ANSWER, "input_tokens": 50, "output_tokens": 10},
    ]
    usage = _trajectory_usage(steps)
    assert usage.input_tokens == 150
    assert usage.output_tokens == 30
    assert usage.total_tokens == 180


def test_trajectory_usage_empty_is_zero():
    usage = _trajectory_usage([])
    assert usage.total_tokens == 0


def test_solver_sets_output_and_attaches_trajectory(monkeypatch):
    traj = Trajectory(question="Q", model="m")
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search"}))
    traj.add(Step(kind=FINAL_ANSWER, content="A"))
    traj.answer = "A"
    fake = wiki_agent.AgentResult(answer="A", trajectory=traj, steps=2)

    monkeypatch.setattr(wiki_agent, "run", lambda q, **kw: fake)

    solve = wiki_agent_solver()
    state = SimpleNamespace(input_text="Q", metadata={}, output=None)
    asyncio.run(solve(state, None))

    assert state.output.completion == "A"
    assert state.metadata["steps"] == 2
    assert state.metadata["trajectory"]["answer"] == "A"
