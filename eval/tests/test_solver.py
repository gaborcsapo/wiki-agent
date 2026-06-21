"""Unit test for the agent-wrapping solver, using a fake agent (no API calls)."""

import asyncio
from types import SimpleNamespace

import wiki_agent
from wiki_agent.trajectory import FINAL_ANSWER, TOOL_CALL, Step, Trajectory

from wiki_eval.solver import wiki_agent_solver


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
