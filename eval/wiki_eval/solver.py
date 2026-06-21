"""Inspect solver that wraps the Wikipedia agent.

This is the one place the eval depends on the agent. It calls the agent's single
public entrypoint, `wiki_agent.run`, and adapts the result into Inspect's model:
the answer becomes `state.output`, and the full trajectory is attached to
`state.metadata` so it shows up in the Inspect transcript / `inspect view`.
"""

from __future__ import annotations

import anyio
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Generate, TaskState, solver

import wiki_agent

from .config import AGENT_MODEL


@solver
def wiki_agent_solver(model: str | None = AGENT_MODEL, max_steps: int = 6):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # The agent loop is synchronous (blocking HTTP + API calls); run it in a
        # worker thread so we don't block Inspect's event loop.
        result = await anyio.to_thread.run_sync(
            lambda: wiki_agent.run(state.input_text, model=model, max_steps=max_steps)
        )
        state.output = ModelOutput.from_content("wiki-agent", result.answer)
        state.metadata["trajectory"] = result.trajectory.to_dict()
        state.metadata["steps"] = result.steps
        return state

    return solve
