"""Scorers for the Wikipedia agent.

Two kinds, illustrating the two grading styles from Anthropic's eval guidance:

* `correctness_judge` — LLM-as-judge (model-based grading) for the open-ended
  correctness of the answer. This is the primary metric.
* `used_wikipedia_tool` — a code-based / trajectory scorer that checks the agent
  actually used its tool rather than answering from memory.

To add a benchmark-specific metric, write another `@scorer` here (e.g.
groundedness of the answer against the fetched extract) and add it to the task's
scorer list.
"""

from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    model_graded_qa,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .config import JUDGE_MODEL


def correctness_judge():
    """LLM-as-judge correctness scorer. The sample `target` is the rubric."""
    return model_graded_qa(model=JUDGE_MODEL, partial_credit=False)


@scorer(metrics=[accuracy(), stderr()])
def used_wikipedia_tool():
    """Code-based scorer: did the agent actually call the wikipedia tool?

    Reads the trajectory the solver attached to `state.metadata`.
    """

    async def score(state: TaskState, target: Target) -> Score:
        steps = state.metadata.get("trajectory", {}).get("steps", [])
        used = any(step.get("kind") == "tool_call" for step in steps)
        return Score(
            value=CORRECT if used else INCORRECT,
            explanation="agent called the wikipedia tool" if used else "no tool call recorded",
        )

    return score
