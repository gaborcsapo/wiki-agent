# Demo Mode & Live-Animated Trajectory Rendering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `wiki-agent demo` command that replays curated cached trajectories with animated, step-by-step rendering (no API key needed), and make ordinary live runs render each step as it happens.

**Architecture:** A single step-at-a-time renderer (`cli._render_step`) is driven two ways. Live runs pass it to `agent.run()` via a new optional `on_step` callback, so panels appear as each step is recorded. Demo runs load committed trajectory JSON, pick one at random, and feed its steps to the same renderer with fixed `time.sleep()` delays. The agent never imports the CLI or `eval`; it only invokes a callback it is handed, preserving the project's one-way boundary.

**Tech Stack:** Python 3.12, `click` (CLI), `rich` (panels), `anthropic` (only at record time), `pytest` (tests).

## Global Constraints

- **No new cross-project imports.** The agent must never import from `eval/`. The only allowed coupling stays `eval → wiki_agent.run`. (CLAUDE.md)
- **`run() -> AgentResult` public surface stays back-compatible.** The new `on_step` param is optional and defaults to `None`; `eval` and existing tests call `run()` unchanged. (CLAUDE.md / spec)
- **All custom logic unit-tested with fakes/monkeypatch. No live API or network calls in tests, and no real `time.sleep`.** Tests must pass with no `ANTHROPIC_API_KEY`. (CLAUDE.md)
- **Config/constants belong in `config.py`**; don't hardcode model ids elsewhere. (CLAUDE.md)
- **Commit scope:** stage only the paths you changed (no `git add -A` at repo root); never stage `.env`. (CLAUDE.md)
- All work is in the `agent/` subproject; run commands from `agent/`.

---

### Task 1: Trajectory/Step deserialization (`from_dict`)

The demo loader reads saved trajectory JSON back into objects, so `Trajectory` needs a `from_dict` mirroring the existing `to_dict`. Pure, no I/O.

**Files:**
- Modify: `agent/wiki_agent/trajectory.py`
- Test: `agent/tests/test_trajectory.py`

**Interfaces:**
- Consumes: existing `Trajectory.to_dict()`, `Step` dataclass.
- Produces: `Trajectory.from_dict(data: dict[str, Any]) -> Trajectory` (classmethod). Round-trips with `to_dict()`.

- [ ] **Step 1: Write the failing test**

Add to `agent/tests/test_trajectory.py`:

```python
def test_from_dict_round_trips_to_dict():
    from wiki_agent.trajectory import ASSISTANT_TEXT, TOOL_RESULT

    traj = Trajectory(question="Q?", model="claude-haiku-4-5")
    traj.add(Step(kind=ASSISTANT_TEXT, content="thinking", input_tokens=10, output_tokens=5))
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search", "query": "q"}))
    traj.add(Step(kind=TOOL_RESULT, content="result text", tool_name="wikipedia"))
    traj.add(Step(kind=FINAL_ANSWER, content="A."))
    traj.answer = "A."

    restored = Trajectory.from_dict(traj.to_dict())

    assert restored == traj
    assert restored.steps[1].tool_input == {"action": "search", "query": "q"}
    assert restored.steps[0].output_tokens == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_trajectory.py::test_from_dict_round_trips_to_dict -v`
Expected: FAIL with `AttributeError: type object 'Trajectory' has no attribute 'from_dict'`.

- [ ] **Step 3: Write minimal implementation**

In `agent/wiki_agent/trajectory.py`, add a classmethod to `Trajectory` (place it right after `to_dict`):

```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        """Rebuild a Trajectory from the dict produced by ``to_dict``."""
        steps = [Step(**step) for step in data.get("steps", [])]
        return cls(
            question=data["question"],
            model=data["model"],
            steps=steps,
            answer=data.get("answer", ""),
        )
```

(`Step(**step)` works because `to_dict` emits exactly the `Step` field names. `@dataclass` gives `Trajectory`/`Step` value equality, so `restored == traj` holds.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: PASS (all trajectory tests).

- [ ] **Step 5: Commit**

```bash
git add agent/wiki_agent/trajectory.py agent/tests/test_trajectory.py
git commit -m "Add Trajectory.from_dict for loading saved trajectories

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `on_step` callback in `agent.run()`

Let callers observe each step as it is recorded, enabling live-animated rendering without changing the public return type.

**Files:**
- Modify: `agent/wiki_agent/agent.py`
- Test: `agent/tests/test_agent.py`

**Interfaces:**
- Consumes: existing `FakeClient` test helper in `test_agent.py`.
- Produces: `run(question, *, model=None, max_steps=..., client=None, on_step: Callable[[Step], None] | None = None) -> AgentResult`. `on_step` is invoked exactly once per recorded `Step`, in order, including the `FINAL_ANSWER` step. When `on_step` is `None`, behavior is identical to today.

- [ ] **Step 1: Write the failing test**

Add to `agent/tests/test_agent.py`:

```python
def test_on_step_callback_fires_per_recorded_step():
    client = FakeClient(
        [
            _response([_tool_block("wikipedia", {"action": "search", "query": "moon"})], "tool_use"),
            _response([_text_block("Neil Armstrong.")], "end_turn"),
        ]
    )
    seen = []
    result = agent.run("Who walked on the moon?", client=client, on_step=seen.append)

    # Same steps the trajectory recorded, in the same order.
    assert [s.kind for s in seen] == [s.kind for s in result.trajectory.steps]
    assert seen[-1].kind == FINAL_ANSWER
    # The objects handed to the callback ARE the recorded steps.
    assert seen == result.trajectory.steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_on_step_callback_fires_per_recorded_step -v`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'on_step'`.

- [ ] **Step 3: Write minimal implementation**

In `agent/wiki_agent/agent.py`:

1. Add the import at the top (after `from dataclasses import dataclass`):

```python
from typing import Callable
```

2. Add `on_step` to the `run` signature:

```python
def run(
    question: str,
    *,
    model: str | None = None,
    max_steps: int = config.DEFAULT_MAX_STEPS,
    client: anthropic.Anthropic | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> AgentResult:
```

3. Immediately after `traj = Trajectory(question=question, model=model)`, add a local emitter:

```python
    def emit(step: Step) -> None:
        traj.add(step)
        if on_step is not None:
            on_step(step)
```

4. Replace every `traj.add(...)` call in the function body with `emit(...)`. There are five: the assistant-text step, the success-path `FINAL_ANSWER` step, the `TOOL_CALL` step, the `TOOL_RESULT` step, and the step-cap `FINAL_ANSWER` step. (Leave the `emit` helper's own `traj.add` as-is.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: PASS — the new test and all existing agent tests (which call `run()` without `on_step`).

- [ ] **Step 5: Commit**

```bash
git add agent/wiki_agent/agent.py agent/tests/test_agent.py
git commit -m "Add optional on_step callback to agent.run for live rendering

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Demo questions + loader/player

The `demos` package holds the curated questions, loads committed trajectory JSON, picks one at random, and plays its steps through a caller-supplied renderer with fixed delays. The player imports neither `cli` nor `eval` — it takes `render_step` and `sleep` as parameters, which also makes it testable with no real sleeping.

**Files:**
- Create: `agent/wiki_agent/demos/__init__.py`
- Create: `agent/wiki_agent/demos/questions.py`
- Create: `agent/wiki_agent/demos/player.py`
- Test: `agent/tests/test_demos.py`

**Interfaces:**
- Consumes: `Trajectory.from_dict` (Task 1), `trajectory.FINAL_ANSWER`.
- Produces:
  - `questions.QUESTIONS: list[str]` (the 10 demo questions).
  - `player.DEMOS_DIR: Path` (the package dir where `*.json` demos live).
  - `player.STEP_DELAY_SECONDS: float`.
  - `player.load_demos(directory: Path = DEMOS_DIR) -> list[Trajectory]` — parse every `*.json` in `directory`, sorted by filename.
  - `player.pick_demo(trajectories: list[Trajectory], rng=random) -> Trajectory` — `rng.choice`.
  - `player.play(traj, render_step, *, sleep=time.sleep, delay=STEP_DELAY_SECONDS) -> None` — for each non-final step, `sleep(delay)` then `render_step(step)`.

- [ ] **Step 1: Create the package and questions**

Create `agent/wiki_agent/demos/__init__.py`:

```python
"""Demo mode: replay curated cached trajectories without an API key."""
```

Create `agent/wiki_agent/demos/questions.py`:

```python
"""The curated demo questions.

These are deliberately hard / multi-hop so the demo shows the agent chaining
several Wikipedia lookups. This list is the single source of truth the recorder
(`record.py`) runs to (re)generate the cached trajectory JSON files.
"""

QUESTIONS = [
    "Who succeeded the English monarch who reigned during the Great Fire of London?",
    "What is the capital of the country that hosted the 1992 Summer Olympics?",
    "Which planet did Voyager 1 fly by first after its launch?",
    "In which city was the architect of the Sagrada Família born?",
    "What is the longest river on the continent where Mount Kilimanjaro is located?",
    "Which chemical element is named after the scientist who created the periodic table?",
    "What is the highest mountain in the country where the Eiffel Tower is located?",
    "In which city was the lead actor of the 1997 film Titanic born?",
    "Which U.S. president signed the law that created the National Park Service?",
    "Which country has won the most FIFA World Cup titles, and in what year was its first win?",
]
```

- [ ] **Step 2: Write the failing test**

Create `agent/tests/test_demos.py`:

```python
"""Unit tests for demo loading and playback — no network, no real sleep."""

import json

from wiki_agent.demos import player
from wiki_agent.trajectory import (
    ASSISTANT_TEXT,
    FINAL_ANSWER,
    TOOL_CALL,
    Step,
    Trajectory,
)


def _sample_trajectory(question: str) -> Trajectory:
    traj = Trajectory(question=question, model="claude-haiku-4-5")
    traj.add(Step(kind=ASSISTANT_TEXT, content="let me look this up"))
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search", "query": "x"}))
    traj.add(Step(kind=FINAL_ANSWER, content="the answer"))
    traj.answer = "the answer"
    return traj


def _write_demo(directory, name, traj):
    (directory / name).write_text(json.dumps(traj.to_dict(), ensure_ascii=False))


def test_load_demos_parses_all_json_sorted(tmp_path):
    _write_demo(tmp_path, "01.json", _sample_trajectory("Q1?"))
    _write_demo(tmp_path, "00.json", _sample_trajectory("Q0?"))

    loaded = player.load_demos(tmp_path)

    assert [t.question for t in loaded] == ["Q0?", "Q1?"]
    assert isinstance(loaded[0], Trajectory)


def test_pick_demo_uses_injected_rng():
    trajs = [_sample_trajectory("A?"), _sample_trajectory("B?")]

    class FakeRng:
        def choice(self, seq):
            return seq[-1]

    assert player.pick_demo(trajs, rng=FakeRng()).question == "B?"


def test_play_renders_non_final_steps_with_injected_sleep():
    traj = _sample_trajectory("Q?")
    rendered = []
    slept = []

    player.play(traj, rendered.append, sleep=slept.append, delay=0.5)

    # FINAL_ANSWER is skipped; the two preceding steps are rendered in order.
    assert [s.kind for s in rendered] == [ASSISTANT_TEXT, TOOL_CALL]
    # One sleep per rendered step, using the supplied delay; no real sleeping.
    assert slept == [0.5, 0.5]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_demos.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` for `player.load_demos`.

- [ ] **Step 4: Write minimal implementation**

Create `agent/wiki_agent/demos/player.py`:

```python
"""Load cached demo trajectories and play them back step by step.

Pure of any rendering or API concerns: ``play`` takes the renderer and the
sleep function as parameters, so the CLI supplies the real ones and tests
supply fakes (no network, no real sleeping).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

from ..trajectory import FINAL_ANSWER, Step, Trajectory

# Cached trajectory JSONs live alongside this module.
DEMOS_DIR = Path(__file__).resolve().parent

# Fixed pause before revealing each step, to simulate a live run.
STEP_DELAY_SECONDS = 0.8


def load_demos(directory: Path = DEMOS_DIR) -> list[Trajectory]:
    """Parse every ``*.json`` demo in ``directory``, sorted by filename."""
    return [
        Trajectory.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    ]


def pick_demo(trajectories: list[Trajectory], rng=random) -> Trajectory:
    """Pick one trajectory at random (``rng`` injectable for tests)."""
    return rng.choice(trajectories)


def play(
    traj: Trajectory,
    render_step: Callable[[Step], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = STEP_DELAY_SECONDS,
) -> None:
    """Reveal each non-final step through ``render_step`` with a pause before each."""
    for step in traj.steps:
        if step.kind == FINAL_ANSWER:
            continue  # the final answer panel is printed by the caller
        sleep(delay)
        render_step(step)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_demos.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add agent/wiki_agent/demos/__init__.py agent/wiki_agent/demos/questions.py agent/wiki_agent/demos/player.py agent/tests/test_demos.py
git commit -m "Add demo questions and trajectory loader/player

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Demo recorder

A function that runs the curated questions through `run()` and writes one JSON per question. The `run` function is injectable so the test uses the existing `FakeClient` — no live API.

**Files:**
- Create: `agent/wiki_agent/demos/record.py`
- Test: `agent/tests/test_demos.py` (add to the existing file)

**Interfaces:**
- Consumes: `agent.run` (Task 2 signature), `questions.QUESTIONS`, `player.DEMOS_DIR`, `Trajectory.to_dict`.
- Produces: `record.record_demos(directory: Path = DEMOS_DIR, *, questions=QUESTIONS, run_fn=run) -> list[Path]` — writes `00.json … NN.json` (zero-padded index) and returns the written paths.

- [ ] **Step 1: Write the failing test**

Add to `agent/tests/test_demos.py`:

```python
def test_record_demos_writes_one_json_per_question(tmp_path):
    from wiki_agent.demos import record
    from wiki_agent.agent import AgentResult

    def fake_run(question):
        return AgentResult(answer="A.", trajectory=_sample_trajectory(question), steps=1)

    paths = record.record_demos(
        tmp_path, questions=["Q-one?", "Q-two?"], run_fn=fake_run
    )

    assert [p.name for p in paths] == ["00.json", "01.json"]
    data = json.loads((tmp_path / "00.json").read_text())
    assert data["question"] == "Q-one?"
    assert data["steps"][0]["kind"] == "assistant_text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_demos.py::test_record_demos_writes_one_json_per_question -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_agent.demos.record'`.

- [ ] **Step 3: Write minimal implementation**

Create `agent/wiki_agent/demos/record.py`:

```python
"""Record the curated demo trajectories.

This is the only demo component that needs an ``ANTHROPIC_API_KEY`` and network
access: it runs each question through the real agent and saves the resulting
trajectory JSON. The CLI exposes it as ``wiki-agent demo --record``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import run
from .player import DEMOS_DIR
from .questions import QUESTIONS


def record_demos(directory: Path = DEMOS_DIR, *, questions=QUESTIONS, run_fn=run) -> list[Path]:
    """Run each question through ``run_fn`` and write ``NN.json`` files.

    Returns the list of written paths. Stable, zero-padded filenames mean a
    re-record overwrites the previous set in place.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, question in enumerate(questions):
        result = run_fn(question)
        path = directory / f"{index:02d}.json"
        path.write_text(
            json.dumps(result.trajectory.to_dict(), indent=2, ensure_ascii=False)
        )
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_demos.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent/wiki_agent/demos/record.py agent/tests/test_demos.py
git commit -m "Add demo recorder to (re)generate cached trajectories

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CLI — step renderer, default-command group, `ask` + `demo`

Refactor rendering into a single-step function, restructure the CLI into a `click` group whose default command is `ask` (so `wiki-agent "Q"` is unchanged), wire live rendering through `on_step`, and add the `demo` command. Update docs in the same commit.

**Files:**
- Modify: `agent/wiki_agent/cli.py`
- Modify: `agent/pyproject.toml` (entry point target)
- Modify: `agent/README.md`, `CLAUDE.md`
- Test: `agent/tests/test_cli.py` (new)

**Interfaces:**
- Consumes: `agent.run(..., on_step=...)` (Task 2), `demos.player.load_demos/pick_demo/play` (Task 3), `demos.record.record_demos` (Task 4).
- Produces: a `click.Group` named `cli` (the new entry point) with commands `ask` (default) and `demo`; `_render_step(step)`, `_render_question(question)`, `_render_final(answer)` helpers.

- [ ] **Step 1: Write the failing test**

Create `agent/tests/test_cli.py`:

```python
"""CLI routing tests using click's CliRunner — no network, no real sleep."""

from click.testing import CliRunner

from wiki_agent import cli
from wiki_agent.agent import AgentResult
from wiki_agent.trajectory import FINAL_ANSWER, Step, Trajectory


def _traj(question="Q?"):
    t = Trajectory(question=question, model="m")
    t.add(Step(kind=FINAL_ANSWER, content="the answer"))
    t.answer = "the answer"
    return t


def test_bare_question_routes_to_ask(monkeypatch):
    captured = {}

    def fake_run(question, *, model=None, max_steps=6, on_step=None):
        captured["question"] = question
        # Drive the live renderer the way the real run would.
        for step in _traj(question).steps:
            if on_step:
                on_step(step)
        return AgentResult(answer="the answer", trajectory=_traj(question), steps=1)

    monkeypatch.setattr(cli, "run", fake_run)
    result = CliRunner().invoke(cli.cli, ["Who walked on the moon?", "--no-save"])

    assert result.exit_code == 0, result.output
    assert captured["question"] == "Who walked on the moon?"
    assert "the answer" in result.output


def test_demo_plays_random_cached_trajectory(monkeypatch):
    monkeypatch.setattr(cli, "load_demos", lambda: [_traj("Demo Q?")])
    monkeypatch.setattr(cli, "pick_demo", lambda trajs: trajs[0])
    # Make playback instant by stubbing the sleep used inside play().
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    result = CliRunner().invoke(cli.cli, ["demo"])

    assert result.exit_code == 0, result.output
    assert "Demo Q?" in result.output
    assert "the answer" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `cli` has no attribute `cli` (group not defined yet).

- [ ] **Step 3: Rewrite `cli.py`**

Replace the entire contents of `agent/wiki_agent/cli.py` with:

```python
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

from . import config
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
def ask(question: str, model: str | None, max_steps: int, save: bool) -> None:
    """Answer QUESTION live, rendering each step as it happens."""
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
```

- [ ] **Step 4: Point the entry point at the group**

In `agent/pyproject.toml`, change the script target:

```toml
[project.scripts]
wiki-agent = "wiki_agent.cli:cli"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: PASS (all tests across the agent subproject).

- [ ] **Step 7: Update docs**

In `agent/README.md`, under the layout table add a row:

```markdown
| `wiki_agent/demos/` | Demo mode: curated questions, cached trajectory JSONs, loader/player/recorder |
```

And in the `## Run` section, append:

```markdown
Replay a curated hard question with animated pacing — no API key required:

```bash
uv run wiki-agent demo
```

Live runs now render each reasoning step, tool call, and result as it happens.
Re-record the cached demos (needs an API key): `uv run wiki-agent demo --record`.
```

In `CLAUDE.md`, in the file-map table add:

```markdown
| `agent/wiki_agent/demos/` | Demo mode: `questions.py`, cached `*.json` trajectories, `player.py` (load/pick/play), `record.py` |
```

And in the `## Commands` agent block, add after the existing `uv run wiki-agent` line:

```bash
uv run wiki-agent demo                 # replay a cached hard question (no API key)
uv run wiki-agent demo --record        # re-record cached demos (needs API key)
```

- [ ] **Step 8: Commit**

```bash
git add agent/wiki_agent/cli.py agent/pyproject.toml agent/tests/test_cli.py agent/README.md CLAUDE.md
git commit -m "Add demo subcommand and live step-by-step CLI rendering

Restructure the CLI into a click group whose default command is ask, so
'wiki-agent \"Q\"' is unchanged. ask now streams each step via on_step;
demo replays a random cached trajectory with animated pacing.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Record the real cached demos

Generate the committed demo JSONs by running the curated questions through the live agent. This needs `ANTHROPIC_API_KEY` (from `agent/.env`) and network access — it is a one-time data-generation step, not covered by tests.

**Files:**
- Create: `agent/wiki_agent/demos/00.json … 09.json` (generated)

- [ ] **Step 1: Record the demos**

Run from `agent/`:

```bash
uv run wiki-agent demo --record
```

Expected: `Recorded 10 demo trajectories.` and ten files `agent/wiki_agent/demos/00.json … 09.json`.

If no API key/network is available, stop here and report that this step must be run by the user with a key in `agent/.env`; the rest of the feature is complete and the `demo` command will print a clear "record them first" message until the JSONs exist.

- [ ] **Step 2: Sanity-check one playback**

Run: `uv run wiki-agent demo`
Expected: a question panel, several step panels appearing with a brief pause between each, then a final-answer panel. Run it a few times to confirm different questions appear.

- [ ] **Step 3: Verify a recorded answer is sensible**

Open one or two of the generated JSONs and confirm the `answer` field actually answers the question (e.g. the Great Fire question resolves to "James II"). If any question consistently fails or the agent can't answer it within the step cap, replace that entry in `questions.py`, re-run `--record`, and re-check.

- [ ] **Step 4: Commit the demo data**

```bash
git add agent/wiki_agent/demos/00.json agent/wiki_agent/demos/01.json agent/wiki_agent/demos/02.json agent/wiki_agent/demos/03.json agent/wiki_agent/demos/04.json agent/wiki_agent/demos/05.json agent/wiki_agent/demos/06.json agent/wiki_agent/demos/07.json agent/wiki_agent/demos/08.json agent/wiki_agent/demos/09.json
git commit -m "Record cached demo trajectories for demo mode

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Demo subcommand (random replay, no key) → Tasks 3, 5, 6. ✅
- Live step-by-step rendering → Tasks 2, 5. ✅
- `on_step` callback seam preserving one-way boundary → Task 2 (player/recorder import only `agent`/`trajectory`, never `cli`/`eval`). ✅
- Committed JSON + regen script → Tasks 4 (recorder), 6 (run it). ✅
- 10 hard/multi-hop questions in code → Task 3 `questions.py`. ✅
- `from_dict` for loading → Task 1. ✅
- Fixed delays, injectable sleep, no `--speed` flag → Task 3 `play`. ✅
- `ask` default command keeps `wiki-agent "Q"` working → Task 5 `_DefaultGroup`. ✅
- Tests: no live API, no network, no real sleep, pass without key → all tests use FakeClient / fixtures / injected sleep / monkeypatched `time.sleep`. ✅
- Docs updated same commit as feature → Task 5 Step 7. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases" — every code step shows complete code. ✅

**Type consistency:** `load_demos`/`pick_demo`/`play` signatures match between Task 3 definition and Task 5 usage. `run(..., on_step=...)` matches between Task 2 and Task 5/test. `record_demos(directory, *, questions, run_fn)` matches Task 4 and the CLI's no-arg `record_demos()` call (all params defaulted). `Trajectory.from_dict` matches Task 1 and `load_demos`. ✅

**Known limitation (documented):** With the default-command group, options placed *before* the question (e.g. `wiki-agent --max-steps 8 "Q"`) won't route; put the question first (`wiki-agent "Q" --max-steps 8`) or use `wiki-agent ask ...`. Documented examples put the question first, matching existing README usage.
