# Wikipedia Agent + Evaluation Suite

Two small, independent systems:

1. **`agent/`** — a minimal, from-scratch agent (Karpathy-style) that answers
   questions by looping over the Claude API with a single Wikipedia tool, and
   records its trajectory for debugging. Driven by a terminal CLI.
2. **`eval/`** — an isolated [Inspect AI](https://inspect.aisi.org.uk/)
   evaluation suite that scores the agent on a custom benchmark using an
   LLM-as-judge, structured so new benchmarks and metrics are easy to add.

## Decoupling

The two are deliberately decoupled — separate folders, separate dependencies,
each runnable on its own. The **only** link is that the eval imports the agent's
single public function:

```
eval ──imports──> wiki_agent.run(question) -> AgentResult ──> agent
```

The agent has zero knowledge of the eval.

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Agent
cd agent && uv sync
uv run wiki-agent "Who was the first person to walk on the Moon?"

# Eval
cd ../eval && uv sync
uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
uv run inspect view start --host 0.0.0.0 --port 7575   # results UI (Tailscale-accessible)
```

Both start on **Haiku**; switch to **Sonnet** via the `*_MODEL` constants in
`agent/wiki_agent/config.py` and `eval/wiki_eval/config.py`.

See `agent/README.md` and `eval/README.md` for details.

## Tests

```bash
cd agent && uv run pytest    # tool parsers, trajectory, loop logic (no API key)
cd eval  && uv run pytest    # solver adapter, custom scorer (no API key)
```
