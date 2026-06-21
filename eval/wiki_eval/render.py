"""Render a trajectory dict as a human-readable markdown dump.

Pure presentation logic for the Inspect UI: the solver feeds the dict it already
attaches to sample metadata, and we turn it into a markdown string surfaced as a
transcript Info entry. No I/O, no Inspect imports — unit-testable offline.
"""

from __future__ import annotations

from typing import Any

# Tool results (e.g. article extracts) can be long; cap them so the transcript
# view stays readable.
MAX_RESULT_CHARS = 1500


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + "\n… (truncated)"


def _format_args(tool_input: dict[str, Any] | None) -> str:
    if not tool_input:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in tool_input.items())


def _format_step(step: dict[str, Any]) -> str | None:
    kind = step.get("kind")
    content = step.get("content") or ""
    if kind == "assistant_text":
        return content.strip() or None
    if kind == "tool_call":
        name = step.get("tool_name") or "tool"
        return f"🔧 **{name}**({_format_args(step.get('tool_input'))})"
    if kind == "tool_result":
        return f"↩️ result:\n```\n{_truncate(content)}\n```"
    if kind == "final_answer":
        return f"**Final answer:** {content.strip()}"
    return None  # unknown kind: skip rather than raise


def format_trajectory(traj: dict[str, Any]) -> str:
    """Return a markdown dump of every assistant output and tool used."""
    lines: list[str] = []
    question = traj.get("question")
    model = traj.get("model")
    if question:
        lines.append(f"**Question:** {question}")
    if model:
        lines.append(f"**Model:** {model}")
    if lines:
        lines.append("")

    for step in traj.get("steps", []):
        rendered = _format_step(step)
        if rendered:
            lines.append(rendered)
            lines.append("")

    answer = traj.get("answer")
    if answer:
        lines.append(f"**Answer:** {answer}")

    return "\n".join(lines).strip()
