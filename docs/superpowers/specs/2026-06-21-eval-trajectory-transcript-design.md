# Readable trajectory dump in the Inspect UI

Date: 2026-06-21
Status: approved (design), pending implementation

## Problem

The eval solver attaches the agent's structured trajectory to
`state.metadata["trajectory"]`. Inspect renders that as nested JSON in the
sample's **Metadata** tab — complete, but hard to read when you just want to see
what the agent said, which tools it called, and what came back.

We want a simple, human-readable dump of the run (all assistant outputs + tools
used) visible in the Inspect UI for a given sample.

## Approach

Additive, lowest-risk (chosen over removing the raw dict, which several scorers
depend on):

1. Keep `state.metadata["trajectory"]` exactly as-is — `used_wikipedia_tool`,
   `retrieval_grounding`, etc. continue to read it.
2. Add a second, human-readable rendering surfaced as an **Info entry in the
   sample transcript** via Inspect's `transcript().info(...)` API. Info events
   render inline in the transcript timeline; a markdown string renders as
   markdown.

## Components

### 1. Pure formatter — `eval/wiki_eval/render.py`

```
format_trajectory(traj: dict) -> str
```

- Input: the dict produced by `Trajectory.to_dict()` (the same one already in
  metadata) — `{"question", "model", "steps": [...], "answer"}`.
- Output: a readable markdown string. For each step, by `kind`:
  - `assistant_text` → the text block.
  - `tool_call` → `🔧 **{tool_name}**(key="value", ...)` from `tool_input`.
  - `tool_result` → the content, truncated to a fixed `MAX_RESULT_CHARS`
    (e.g. 1500) with a `… (truncated)` marker so a long article extract doesn't
    flood the view.
  - `final_answer` → rendered under a **Final answer** heading.
- Pure: no network, no API key, no Inspect imports. Operates only on the dict.
- Defensive: unknown `kind` values and missing fields are skipped/handled, never
  raise (mirrors the tool's "never raise into the model" ethos).

### 2. Solver wiring — `eval/wiki_eval/solver.py`

After building the trajectory dict, emit the readable dump:

```python
from inspect_ai.log import transcript
from .render import format_trajectory

traj = result.trajectory.to_dict()
state.metadata["trajectory"] = traj
state.metadata["steps"] = result.steps
transcript().info(format_trajectory(traj))
```

One import + one call. No change to the agent, no new cross-project coupling
(we operate on the dict the agent already returns).

## Testing

- **Unit (offline, no API key):** `eval/tests` test for `format_trajectory` built
  from a hand-constructed trajectory dict covering each step kind plus an
  over-long tool result (asserts truncation) and an unknown kind (asserts no
  raise). Per CLAUDE.md: no live API/network in tests.
- **Live smoke test:** run a 2-sample slice of `factual_qa` on default Haiku and
  confirm the Info entry renders in `inspect view`:

  ```bash
  cd eval
  uv run inspect eval wiki_eval/tasks.py@factual_qa \
      --model anthropic/claude-haiku-4-5 --limit 2
  uv run inspect view start --host 0.0.0.0 --port 7575
  ```

  (Requires `ANTHROPIC_API_KEY` in `eval/.env`.)

## Out of scope

- Removing or restructuring `metadata["trajectory"]`.
- Any change to the agent subproject.
- Rich/HTML rendering beyond markdown.
