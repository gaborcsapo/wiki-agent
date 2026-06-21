"""The agent: a minimal, hand-written loop over the Claude API.

No SDK tool-runner helper is used — the manual loop is the point. The agent
sends the question to Claude with the Wikipedia tool available, executes any
tool calls itself, feeds results back, and repeats until Claude answers (or a
step cap is hit). Every turn is recorded in a Trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import anthropic

from . import config, wikipedia
from .trajectory import (
    ASSISTANT_TEXT,
    FINAL_ANSWER,
    TOOL_CALL,
    TOOL_RESULT,
    Step,
    Trajectory,
)

SYSTEM_PROMPT = (
    "You are a research assistant answering questions that often require "
    "combining facts from several Wikipedia articles.\n\n"
    "Work efficiently:\n"
    "- Break the question into the specific facts you need to find.\n"
    "- Use the `wikipedia` tool: search with precise terms, then read the most "
    "relevant article. Prefer the single best article over many tangential ones.\n"
    "- When you need several independent lookups, request them together in one "
    "`wikipedia` call as a `queries` or `titles` list so they run in parallel — "
    "fewer round-trips than one at a time.\n"
    "- Base every fact on what you read, not prior memory.\n\n"
    "Finish decisively:\n"
    "- As soon as you have the needed facts, stop searching and answer. Do not "
    "keep exploring once you can answer.\n"
    "- You have a limited number of tool calls. Always commit to a final answer "
    "before you run out — if you are not fully certain, give your best-supported "
    "answer and briefly note the reasoning. Only say the answer is unavailable if "
    "Wikipedia genuinely lacks it.\n"
    "- State the final answer first, concisely and in plain text (the name, "
    "number, or date). Do not add filler like 'Perfect!' or restate the question."
)


@dataclass
class AgentResult:
    """The single clean return type that the eval suite consumes."""

    answer: str
    trajectory: Trajectory
    steps: int


def _final_text(content: list) -> str:
    """Join the text blocks of an assistant message into one string."""
    return "\n".join(block.text for block in content if block.type == "text").strip()


def run(
    question: str,
    *,
    model: str | None = None,
    max_steps: int = config.DEFAULT_MAX_STEPS,
    client: anthropic.Anthropic | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
    """Answer ``question`` with the Wikipedia agent.

    ``client`` is injectable so tests can pass a fake. ``on_step`` is invoked
    once per recorded step, in order, enabling live step-by-step rendering.
    Returns an AgentResult carrying the final answer and the full trajectory.
    """
    model = model or config.AGENT_MODEL
    client = client or anthropic.Anthropic()
    traj = Trajectory(question=question, model=model)

    def emit(step: Step) -> None:
        traj.add(step)
        if on_step is not None:
            on_step(step)

    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[wikipedia.TOOL_SCHEMA],
            messages=messages,
            # Haiku doesn't support thinking/effort; Sonnet does, but FRAMES
            # tuning found no benefit (docs/hill-climbing-report.md), so we
            # don't enable them.
        )

        text = _final_text(response.content)
        usage = getattr(response, "usage", None)
        if text:
            emit(
                Step(
                    kind=ASSISTANT_TEXT,
                    content=text,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                )
            )

        if response.stop_reason != "tool_use":
            # Claude is done — this turn's text is the final answer.
            traj.answer = text
            emit(Step(kind=FINAL_ANSWER, content=text))
            return AgentResult(answer=text, trajectory=traj, steps=step)

        # Echo the assistant turn (including tool_use blocks) back into history.
        messages.append({"role": "assistant", "content": response.content})

        # Execute every tool call and collect the results.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            emit(Step(kind=TOOL_CALL, tool_name=block.name, tool_input=dict(block.input)))
            result = wikipedia.dispatch(block.input)
            emit(Step(kind=TOOL_RESULT, content=result, tool_name=block.name))
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    # Budget exhausted. Make one final call with NO tools so the model must
    # commit to a best-effort answer from what it has gathered, instead of
    # looping forever or emitting a canned non-answer (which always scores wrong).
    response = client.messages.create(
        model=model,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "You are out of tool calls. Give your best final answer now, "
                    "concisely and in plain text, based on what you have found."
                ),
            }
        ],
    )
    answer = _final_text(response.content) or "I could not find a conclusive answer."
    traj.answer = answer
    emit(Step(kind=FINAL_ANSWER, content=answer))
    return AgentResult(answer=answer, trajectory=traj, steps=max_steps)
