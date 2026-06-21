"""Command-line interface for the Wikipedia agent.

Two commands:

    wiki-agent "Who was the first person to walk on the moon?"   # live run
    wiki-agent demo                                              # cached replay

`ask` (the default command) runs the agent live, rendering each step as it
happens. `demo` replays a random curated trajectory with animated pacing and
needs no API key. Both share one step renderer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import cache, config
from .agent import run
from .demos.player import load_demos, pick_demo, play
from .demos.record import record_demos
from .trajectory import ASSISTANT_TEXT, FINAL_ANSWER, TOOL_CALL, TOOL_RESULT, Step

console = Console()

# Panel styling per step kind.
_STYLE = {
    ASSISTANT_TEXT: ("🧠 Reasoning", "cyan"),
    TOOL_CALL: ("🔧 Tool call", "yellow"),
    TOOL_RESULT: ("📄 Tool result", "green"),
    FINAL_ANSWER: ("✅ Final answer", "bold magenta"),
}

_TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def _render_question(question: str) -> None:
    console.print(Panel(Text(question, style="bold white"), title="❓ Question", border_style="white"))


def _render_step(step: Step) -> None:
    """Render one trajectory step as a panel. Used live and during demo replay."""
    if step.kind == FINAL_ANSWER:
        return  # printed separately by _render_final
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


def _render_final(answer: str) -> None:
    console.print(Panel(Text(answer, style="bold magenta"), title="✅ Final answer", border_style="magenta"))


class _DefaultGroup(click.Group):
    """A group that routes an unknown first argument to a default command,
    so ``wiki-agent "a question"`` still works once subcommands exist."""

    def __init__(self, *args, default_command: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_command = default_command

    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = [self._default_command, *args]
        return super().parse_args(ctx, args)


@click.group(cls=_DefaultGroup, default_command="ask")
def cli() -> None:
    """Answer questions with the Wikipedia agent."""


@cli.command()
@click.argument("question")
@click.option("--model", default=None, help="Override the agent model (default: Haiku).")
@click.option("--max-steps", default=config.DEFAULT_MAX_STEPS, show_default=True, help="Max tool-use iterations.")
@click.option("--save/--no-save", default=True, show_default=True, help="Save the trajectory JSON to traces/.")
@click.option("--no-cache", is_flag=True, help="Bypass the Wikipedia disk cache for this run.")
@click.option("--clear-cache", is_flag=True, help="Delete cached Wikipedia pages before running.")
def ask(question: str, model: str | None, max_steps: int, save: bool,
        no_cache: bool, clear_cache: bool) -> None:
    """Answer QUESTION live, rendering each step as it happens."""
    if clear_cache:
        removed = cache.clear()
        console.print(f"[dim]Cleared {removed} cached Wikipedia entries.[/dim]")
    if no_cache:
        config.CACHE_ENABLED = False
    load_dotenv()
    _render_question(question)
    result = run(question, model=model, max_steps=max_steps, on_step=_render_step)
    _render_final(result.answer)
    if save:
        path = result.trajectory.save(_TRACES_DIR)
        console.print(f"[dim]Trajectory saved to {path}[/dim]")


@cli.command()
@click.option("--record", "do_record", is_flag=True, help="Re-record the cached demos (needs an API key).")
def demo(do_record: bool) -> None:
    """Replay a random curated trajectory with animated pacing (no API key)."""
    if do_record:
        load_dotenv()
        paths = record_demos()
        console.print(f"[dim]Recorded {len(paths)} demo trajectories.[/dim]")
        return

    trajectories = load_demos()
    if not trajectories:
        raise click.ClickException(
            "No cached demos found. Record them first: wiki-agent demo --record"
        )
    traj = pick_demo(trajectories)
    _render_question(traj.question)
    play(traj, _render_step)
    _render_final(traj.answer)


if __name__ == "__main__":
    cli()
