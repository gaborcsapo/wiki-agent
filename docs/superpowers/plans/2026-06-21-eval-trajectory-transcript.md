# Readable Trajectory Dump in the Inspect UI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the agent's per-sample trajectory as a human-readable Info entry in the Inspect transcript UI, alongside the existing raw metadata dict.

**Architecture:** A pure `format_trajectory(dict) -> str` formatter (new `eval/wiki_eval/render.py`) turns the trajectory dict the solver already builds into a markdown string; the solver emits it via Inspect's `transcript().info(...)`. The raw `metadata["trajectory"]` dict is left untouched so existing scorers keep working.

**Tech Stack:** Python, Inspect AI (`inspect_ai.log.transcript`), pytest, uv.

## Global Constraints

- The agent must never import from `eval/`; do not add new cross-project imports. Operate only on the trajectory dict the solver already has.
- Separate pure logic from I/O: the formatter must be importable and testable with no network and no `ANTHROPIC_API_KEY`.
- No live API or network calls in tests.
- Stage only the paths you changed when committing (no `git add -A` / `git add .` at repo root).
- Run eval commands from within `eval/` (its own uv venv).

---

### Task 1: Pure trajectory formatter

**Files:**
- Create: `eval/wiki_eval/render.py`
- Test: `eval/tests/test_render.py`

**Interfaces:**
- Consumes: a trajectory dict shaped like `Trajectory.to_dict()`:
  `{"question": str, "model": str, "answer": str, "steps": [ {"kind": str, "content": str, "tool_name": str|None, "tool_input": dict|None, ...} ]}`.
  Step kinds: `"assistant_text"`, `"tool_call"`, `"tool_result"`, `"final_answer"`.
- Produces: `format_trajectory(traj: dict) -> str` and module constant `MAX_RESULT_CHARS = 1500`.

- [ ] **Step 1: Write the failing test**

```python
# eval/tests/test_render.py
from wiki_eval.render import MAX_RESULT_CHARS, format_trajectory


def _traj(steps, answer="42"):
    return {"question": "Q?", "model": "claude-haiku-4-5", "answer": answer, "steps": steps}


def test_renders_question_model_and_answer():
    out = format_trajectory(_traj([], answer="The answer is 42."))
    assert "Q?" in out
    assert "claude-haiku-4-5" in out
    assert "The answer is 42." in out


def test_renders_assistant_text():
    out = format_trajectory(_traj([{"kind": "assistant_text", "content": "Let me search."}]))
    assert "Let me search." in out


def test_renders_tool_call_with_name_and_args():
    out = format_trajectory(_traj([
        {"kind": "tool_call", "tool_name": "get_article", "tool_input": {"title": "Moon"}},
    ]))
    assert "get_article" in out
    assert "Moon" in out


def test_truncates_long_tool_result():
    big = "x" * (MAX_RESULT_CHARS + 500)
    out = format_trajectory(_traj([{"kind": "tool_result", "content": big}]))
    assert "truncated" in out
    assert out.count("x") <= MAX_RESULT_CHARS


def test_unknown_kind_does_not_raise():
    out = format_trajectory(_traj([{"kind": "mystery", "content": "??"}]))
    assert isinstance(out, str)


def test_missing_fields_do_not_raise():
    out = format_trajectory({"steps": [{"kind": "tool_call"}]})
    assert isinstance(out, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd eval && uv run pytest tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wiki_eval.render'`.

- [ ] **Step 3: Write minimal implementation**

```python
# eval/wiki_eval/render.py
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
    return ", ".join(f'{k}={v!r}' for k, v in tool_input.items())


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd eval && uv run pytest tests/test_render.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/jihgaboot/gabor/anthropic-takehome
git add eval/wiki_eval/render.py eval/tests/test_render.py
git commit -m "feat(eval): pure trajectory-to-markdown formatter"
```

---

### Task 2: Wire the formatter into the solver

**Files:**
- Modify: `eval/wiki_eval/solver.py`

**Interfaces:**
- Consumes: `format_trajectory` from Task 1; `transcript` from `inspect_ai.log`.
- Produces: no new public surface — the solver still returns `TaskState`; it now also emits one transcript Info event per sample.

- [ ] **Step 1: Add the imports**

In `eval/wiki_eval/solver.py`, alongside the existing imports:

```python
from inspect_ai.log import transcript

from .config import AGENT_MODEL
from .render import format_trajectory
```

(Keep the existing `from .config import AGENT_MODEL`; just add the `transcript` and `render` imports.)

- [ ] **Step 2: Emit the readable dump in `solve`**

Replace the body that sets metadata so it reads:

```python
        state.output = ModelOutput.from_content("wiki-agent", result.answer)
        traj = result.trajectory.to_dict()
        state.metadata["trajectory"] = traj
        state.metadata["steps"] = result.steps
        transcript().info(format_trajectory(traj))
        return state
```

- [ ] **Step 3: Verify nothing broke (offline)**

Run: `cd eval && uv run pytest -q`
Expected: PASS (all existing tests + Task 1's still pass; no API key needed).

- [ ] **Step 4: Commit**

```bash
cd /home/jihgaboot/gabor/anthropic-takehome
git add eval/wiki_eval/solver.py
git commit -m "feat(eval): surface trajectory as transcript Info entry"
```

---

### Task 3: Live smoke test (2 samples, Haiku)

**Files:** none (verification only).

- [ ] **Step 1: Confirm an API key is available**

Run: `cd eval && test -f .env && grep -q ANTHROPIC_API_KEY .env && echo "key present" || echo "MISSING: set ANTHROPIC_API_KEY in eval/.env"`
Expected: `key present`. If missing, ask the user to add it before continuing.

- [ ] **Step 2: Run a 2-sample slice on default Haiku**

Run:
```bash
cd eval && uv run inspect eval wiki_eval/tasks.py@factual_qa \
    --model anthropic/claude-haiku-4-5 --limit 2
```
Expected: the eval completes, writing a `.eval` log under `eval/logs/`.

- [ ] **Step 3: Confirm the Info entry is present in the log**

Run:
```bash
cd eval && uv run python -c "
from inspect_ai.log import list_eval_logs, read_eval_log
log = read_eval_log(list_eval_logs()[-1].name)
for s in log.samples:
    kinds = [e.event for e in s.events]
    infos = [e for e in s.events if e.event == 'info']
    print(s.id, 'info_events=', len(infos))
    if infos:
        print(str(infos[0].data)[:300])
"
```
Expected: each sample reports `info_events= 1` and prints the start of the markdown dump (Question/Model/tool calls).

- [ ] **Step 4: (Optional) Eyeball it in the UI**

Run: `cd eval && uv run inspect view start --host 0.0.0.0 --port 7575`
Open a sample → the transcript shows an **Info** entry with the readable dump.

---

## Self-Review

- **Spec coverage:** pure formatter (Task 1) ✓; solver wiring via `transcript().info` keeping the dict (Task 2) ✓; offline unit test + 2-sample Haiku live test (Tasks 1 & 3) ✓; no agent changes / no new coupling ✓.
- **Placeholder scan:** none — all steps carry concrete code/commands.
- **Type consistency:** `format_trajectory(traj: dict) -> str` and `MAX_RESULT_CHARS` are used identically in the test, the implementation, and the solver call.
