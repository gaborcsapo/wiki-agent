"""Trajectory recording: a structured, serializable record of an agent run.

The trajectory is what makes the agent debuggable — every model turn, tool call,
tool result, and the final answer is captured in order and can be saved to JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Step kinds.
ASSISTANT_TEXT = "assistant_text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
FINAL_ANSWER = "final_answer"


@dataclass
class Step:
    """One event in the agent's run."""

    kind: str
    content: str = ""
    # Populated for tool_call steps.
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    # Populated for steps that consumed model tokens.
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Trajectory:
    """The full ordered record of a single agent run."""

    question: str
    model: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Rebuild a Trajectory from the dict produced by ``to_dict``.

        Unknown keys are dropped so a cached demo JSON written by an older/newer
        ``Step`` schema still replays instead of raising ``TypeError``.
        """
        known = {f.name for f in fields(Step)}
        steps = [
            Step(**{k: v for k, v in step.items() if k in known})
            for step in data.get("steps", [])
        ]
        return cls(
            question=data["question"],
            model=data["model"],
            steps=steps,
            answer=data.get("answer", ""),
        )

    def save(self, directory: str | Path) -> Path:
        """Write the trajectory to ``<directory>/<timestamp>.json`` and return the path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = directory / f"{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path
