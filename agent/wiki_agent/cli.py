"""Command-line interface for the Wikipedia agent.

Runs the agent on a question and renders its trajectory as a sequence of
color-coded panels, then prints the final answer.

    wiki-agent "Who was the first person to walk on the moon?"
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import config
from .agent import run
from .trajectory import ASSISTANT_TEXT, FINAL_ANSWER, TOOL_CALL, TOOL_RESULT, Trajectory

console = Console()

# Panel styling per step kind.
_STYLE = {
    ASSISTANT_TEXT: ("🧠 Reasoning", "cyan"),
    TOOL_CALL: ("🔧 Tool call", "yellow"),
    TOOL_RESULT: ("📄 Tool result", "green"),
    FINAL_ANSWER: ("✅ Final answer", "bold magenta"),
}

_TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def _render(traj: Trajectory) -> None:
    console.print(Panel(Text(traj.question, style="bold white"), title="❓ Question", border_style="white"))
    for step in traj.steps:
        if step.kind == FINAL_ANSWER:
            continue  # printed separately at the end
        title, color = _STYLE.get(step.kind, (step.kind, "white"))
        if step.kind == TOOL_CALL:
            body = f"{step.tool_name}({json.dumps(step.tool_input)})"
        else:
            body = step.content
            if len(body) > 1000:
                body = body[:1000] + "\n…(truncated)"
        meta = ""
        if step.output_tokens is not None:
            meta = f"  [dim](in {step.input_tokens} / out {step.output_tokens} tok)[/dim]"
        console.print(Panel(body, title=title + meta, border_style=color))


@click.command()
@click.argument("question")
@click.option("--model", default=None, help="Override the agent model (default: Haiku).")
@click.option("--max-steps", default=config.DEFAULT_MAX_STEPS, show_default=True, help="Max tool-use iterations.")
@click.option("--save/--no-save", default=True, show_default=True, help="Save the trajectory JSON to traces/.")
def main(question: str, model: str | None, max_steps: int, save: bool) -> None:
    """Answer QUESTION with the Wikipedia agent and show its trajectory."""
    # Load ANTHROPIC_API_KEY from a .env file (cwd or a parent) if present.
    load_dotenv()
    result = run(question, model=model, max_steps=max_steps)
    _render(result.trajectory)
    console.print(Panel(Text(result.answer, style="bold magenta"), title="✅ Final answer", border_style="magenta"))
    if save:
        path = result.trajectory.save(_TRACES_DIR)
        console.print(f"[dim]Trajectory saved to {path}[/dim]")


if __name__ == "__main__":
    main()
