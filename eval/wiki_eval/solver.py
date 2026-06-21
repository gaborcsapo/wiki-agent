"""Inspect solver that wraps the Wikipedia agent.

This is the one place the eval depends on the agent. It calls the agent's single
public entrypoint, `wiki_agent.run`, and adapts the result into Inspect's model:
the answer becomes `state.output`, and the full trajectory is attached to
`state.metadata` so it shows up in the Inspect transcript / `inspect view`.

The agent calls Claude directly (it never goes through Inspect's model layer), so
Inspect would otherwise see zero agent token usage. We re-attribute the agent's
own per-step usage back to Inspect (`_record_agent_usage`) so the run's Model and
Tokens columns reflect the agent — not just the judge.
"""

from __future__ import annotations

import anyio
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.solver import Generate, TaskState, solver

import wiki_agent

from .config import AGENT_MODEL


def _trajectory_usage(steps: list[dict]) -> ModelUsage:
    """Sum the agent's per-step token counts into a ModelUsage (pure).

    Only assistant/final-answer turns carry token counts; tool steps don't, and
    a truncated turn may have None — both contribute zero.
    """
    inp = sum(s.get("input_tokens") or 0 for s in steps)
    out = sum(s.get("output_tokens") or 0 for s in steps)
    return ModelUsage(input_tokens=inp, output_tokens=out, total_tokens=inp + out)


def _record_agent_usage(model: str, steps: list[dict]) -> ModelUsage:
    """Attribute the agent's usage to Inspect's per-model accounting.

    Best-effort: this is pure bookkeeping for the `inspect view` Tokens column,
    so it must never break a run. We qualify the model with the `anthropic/`
    provider prefix to match how the run is labelled (Anthropic-only project).
    """
    usage = _trajectory_usage(steps)
    if usage.total_tokens:
        try:
            # Internal API: the public `record_model_usage` only feeds token
            # *limits*, not the per-model stats that drive the Tokens column.
            from inspect_ai.model._model import record_and_check_model_usage

            record_and_check_model_usage(f"anthropic/{model}", usage)
        except Exception:
            pass
    return usage


@solver
def wiki_agent_solver(model: str | None = AGENT_MODEL, max_steps: int = 6):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # The agent loop is synchronous (blocking HTTP + API calls); run it in a
        # worker thread so we don't block Inspect's event loop.
        result = await anyio.to_thread.run_sync(
            lambda: wiki_agent.run(state.input_text, model=model, max_steps=max_steps)
        )
        traj = result.trajectory.to_dict()
        usage = _record_agent_usage(result.trajectory.model, traj.get("steps", []))
        state.output = ModelOutput.from_content("wiki-agent", result.answer)
        state.output.usage = usage
        state.metadata["trajectory"] = traj
        state.metadata["steps"] = result.steps
        return state

    return solve
