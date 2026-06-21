# wiki-agent

A minimal, from-scratch agent that answers questions using the Claude API and a
single Wikipedia tool. Built to show the basic building blocks of an agentic
loop — no framework, just a hand-written loop over `messages.create`.

## What it does

1. Sends your question to Claude with one tool available: `wikipedia`
   (`action=search` to find articles, `action=get_article` to read one).
2. Executes any tool calls against the MediaWiki API and feeds results back.
3. Repeats until Claude produces a final answer (capped by `--max-steps`).
4. Records the full **trajectory** (every reasoning turn, tool call, and result)
   for debugging, and renders it in the terminal.

## Layout

| File | Responsibility |
|------|----------------|
| `wiki_agent/config.py` | Tunable constants (model, endpoint, limits) — one place to swap Haiku → Sonnet |
| `wiki_agent/wikipedia.py` | The single tool: schema, search/get_article, pure parsers separated from HTTP |
| `wiki_agent/trajectory.py` | `Trajectory`/`Step` dataclasses + JSON persistence |
| `wiki_agent/agent.py` | The loop: `run(question) -> AgentResult` |
| `wiki_agent/cli.py` | `wiki-agent` CLI with `rich`-rendered trajectory |

The public API is just `run()` and `AgentResult` — the single clean boundary the
evaluation suite depends on.

## Setup

```bash
cd agent
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
uv run wiki-agent "Who was the first person to walk on the moon?"
uv run wiki-agent "What is the tallest mountain on Earth?" --max-steps 8
uv run wiki-agent "..." --no-save        # don't write a trace file
```

Trajectories are saved to `agent/traces/<timestamp>.json` by default.

## Test

```bash
uv run pytest
```

Tests cover the tool parsers, trajectory serialization, and the loop logic (with
a fake Claude client — no network or API key required).
