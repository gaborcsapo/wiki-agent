# Wikipedia Agent + Evaluation Suite

Two small, independent systems:

1. **`agent/`** — a minimal, from-scratch agent (Karpathy-style) that answers
   questions by looping over the Claude API with a single Wikipedia tool, and
   records its trajectory for debugging. Driven by a terminal CLI.
2. **`eval/`** — an isolated [Inspect AI](https://inspect.aisi.org.uk/)
   evaluation suite that scores the agent on benchmarks using an LLM-as-judge
   plus code-based scorers, structured so new benchmarks and metrics are easy
   to add.

## The one boundary

The two are deliberately decoupled — separate folders, separate dependencies,
each runnable on its own. The **only** link is that the eval imports the agent's
single public function:

```
eval ──imports──> wiki_agent.run(question) -> AgentResult ──> agent
```

The agent has zero knowledge of the eval.

## Where to start

Each subproject is self-contained and documented in its own README. Both need an
Anthropic API key (per-subproject `.env`, gitignored, or `ANTHROPIC_API_KEY`).

```bash
# Agent — answer a question live in the terminal      → details: agent/README.md
cd agent && uv sync && uv run wiki-agent "Who was the first person to walk on the Moon?"

# Eval — score the agent on a benchmark               → details: eval/README.md
cd eval && uv sync && uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
```

- **[`agent/README.md`](agent/README.md)** — how the loop works, the CLI,
  demo mode, the file map, and tests.
- **[`eval/README.md`](eval/README.md)** — the Inspect taxonomy, the available
  benchmarks, running evals, viewing results, and adding your own.
