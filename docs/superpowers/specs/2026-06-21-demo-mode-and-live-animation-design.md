# Demo mode & live-animated trajectory rendering — design

**Date:** 2026-06-21
**Subproject:** `agent/`
**Status:** Approved (ready for implementation plan)

## Problem

Two related gaps in the `wiki-agent` CLI:

1. **No demo mode.** There is no way to show off the agent without an
   `ANTHROPIC_API_KEY` in place. We want a command that replays a curated set of
   hard questions live, so anyone can see how the CLI looks and behaves.
2. **Live runs are not animated.** Today `cli.main` calls `run()`, which executes
   the *entire* loop and builds the full `Trajectory`, and only then does
   `_render()` print every panel at once (`cli.py:60-69`). The user watches
   nothing, then the whole transcript dumps out. Seeing reasoning, tool calls,
   and results appear as they happen would be far nicer.

## Core insight

Both features need the same capability: **render the trajectory one step at a
time** instead of in a single batch. Once rendering is step-at-a-time:

- **Live mode** drives the renderer from the agent loop — each panel appears the
  instant that step completes. Real model/tool latency provides the pacing; no
  artificial sleep is needed.
- **Demo mode** drives the *same* renderer from a cached trajectory, inserting a
  fixed `time.sleep()` between steps to simulate that pacing.

One rendering code path, two sources of steps.

## Approach (chosen: callback seam)

Add an optional `on_step` callback to `agent.run()`. The agent invokes it as each
`Step` is recorded. Consumers supply their own behavior:

- Live CLI passes a "render this step" function.
- Demo replays from JSON and does not use the callback at all.
- `eval` (and existing tests) pass nothing → behavior is **unchanged**.

This preserves the project's one allowed coupling and one-way boundary: the agent
never imports the CLI or `eval`; it only invokes a callback it was handed. The
`run() -> AgentResult` surface that `eval` depends on is unchanged (new param is
optional and defaults to `None`).

Rejected alternatives:

- **Streaming the Anthropic API token-by-token.** True streaming but a much
  larger change to the loop; per-character typewriter was already declined for
  demo. YAGNI for now.
- **Demo-only, leave live runs batch.** Smallest, but discards the live-animation
  win and leaves two separate rendering paths to maintain.

## Components

| Piece | Responsibility |
|------|----------------|
| `cli._render_step(step)` | The body of today's `_render` loop extracted into a single-step function — the one place that knows how a `Step` renders as a panel. |
| `agent.run(..., on_step=None)` | New optional `Callable[[Step], None]`, invoked as each step is recorded (assistant text, tool call, tool result, final answer). |
| `wiki_agent/demos/questions.py` | The 10 hard / multi-hop questions as a `QUESTIONS` list. Lives in code; drives the recorder. |
| `wiki_agent/demos/*.json` | Committed cached trajectories in the existing trajectory JSON format. |
| demos **loader** | Load all demo JSONs → `list[Trajectory]`; pick one at random (RNG injectable for tests). |
| demos **player** | Iterate a trajectory's steps → `_render_step` + a fixed `time.sleep` between steps. Sleep function injectable so tests never sleep. |
| demos **recorder** | Runs `QUESTIONS` through `run()` (needs an API key) and writes the JSON files. The only API-key-needing part. |

## CLI shape

Restructure `cli:main` into a `click.Group` with two commands:

- **`ask`** — current behavior, now rendered live by passing `_render_step` as
  `on_step`. Made the **default command** so `wiki-agent "question"` keeps working
  unchanged (zero breakage to documented usage).
- **`demo`** — load `demos/`, pick one trajectory at random, play it back with
  fixed delays. A hidden `--record` flag re-records the JSONs from `QUESTIONS`
  (keeps questions + file I/O in one module).

### Data flow

- `wiki-agent "Q"` → `ask` → `run(Q, on_step=_render_step)` → panels stream live →
  final answer panel.
- `wiki-agent demo` → load demos/ → random pick → player → `_render_step` per step
  with fixed delays → final answer panel.
- `wiki-agent demo --record` → run the 10 `QUESTIONS` live → save JSONs to
  `demos/`.
- `eval` → `run(Q)` (no callback) → unchanged.

## Trajectory (de)serialization

The demo loader reads trajectory JSON back into objects, so `Trajectory`/`Step`
gain a `from_dict` that mirrors the existing `to_dict`. Small, pure, and
unit-tested. No change to the on-disk format.

## Playback timing

Fixed, hardcoded delays (no `--speed`/`--no-delay` CLI surface). The sleep
function is a parameter of the player so tests inject a no-op. Live mode uses no
artificial delay at all — real latency paces it.

## Testing (per CLAUDE.md: no live API, no network, no real sleep)

- **`on_step` callback** — reuse the existing `FakeClient` pattern in
  `test_agent.py`; assert the callback fires once per recorded step, in order,
  and that omitting it leaves behavior unchanged.
- **Loader** — fixture JSON files → assert parsed into `Trajectory`; seeded RNG →
  deterministic pick.
- **Player** — inject a no-op sleep and a recording fake renderer → assert every
  step is rendered in order and no real sleeping occurs.
- **`from_dict`** — round-trip `to_dict`/`from_dict` equality on a sample
  trajectory.
- **Recorder** — its API call is mocked (`FakeClient`); assert it writes one JSON
  per question.

All tests must pass with no `ANTHROPIC_API_KEY` set.

## Out of scope (YAGNI)

- Token-by-token / typewriter streaming.
- Configurable playback speed flags.
- Selecting a specific demo by name/index (random pick only, as requested).

## Docs

Update `CLAUDE.md` in the same commit as the implementation: new `demo`
subcommand, the `ask` default command, the `demos/` file map entry, and the
recorder command. Update the agent `README.md` usage examples.
