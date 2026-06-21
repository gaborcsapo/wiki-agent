"""The agent: a minimal, hand-written loop over the Claude API.

No SDK tool-runner helper is used — the manual loop is the point. The agent
sends the question to Claude with the Wikipedia tool available, executes any
tool calls itself, feeds results back, and repeats until Claude answers (or a
step cap is hit). Every turn is recorded in a Trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "You are a research assistant. Answer the user's question using the "
    "`wikipedia` tool to look up facts. Search for relevant articles, read the "
    "ones you need, and base your answer on what you find rather than prior "
    "memory. When you have enough information, give a concise final answer in "
    "plain text. If Wikipedia does not contain the answer, say so."
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
) -> AgentResult:
    """Answer ``question`` with the Wikipedia agent.

    ``client`` is injectable so tests can pass a fake. Returns an AgentResult
    carrying the final answer and the full trajectory.
    """
    model = model or config.AGENT_MODEL
    client = client or anthropic.Anthropic()
    traj = Trajectory(question=question, model=model)

    messages: list[dict] = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[wikipedia.TOOL_SCHEMA],
            messages=messages,
            # NOTE: on Haiku we omit thinking/effort (unsupported). When moving
            # to Sonnet, add: thinking={"type": "adaptive"}.
        )

        text = _final_text(response.content)
        usage = getattr(response, "usage", None)
        if text:
            traj.add(
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
            traj.add(Step(kind=FINAL_ANSWER, content=text))
            return AgentResult(answer=text, trajectory=traj, steps=step)

        # Echo the assistant turn (including tool_use blocks) back into history.
        messages.append({"role": "assistant", "content": response.content})

        # Execute every tool call and collect the results.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            traj.add(Step(kind=TOOL_CALL, tool_name=block.name, tool_input=dict(block.input)))
            result = wikipedia.dispatch(block.input)
            traj.add(Step(kind=TOOL_RESULT, content=result, tool_name=block.name))
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    # Step cap reached without a final answer.
    answer = "I could not reach a conclusive answer within the step limit."
    traj.answer = answer
    traj.add(Step(kind=FINAL_ANSWER, content=answer))
    return AgentResult(answer=answer, trajectory=traj, steps=max_steps)
