# CLAUDE.md

Guidance for agents and engineers working in this repo. Keep it current: if you
change a convention or command, update this file in the same commit.

## What this is

Two small, **independent** subprojects:

- **`agent/`** — a from-scratch agent (a hand-written loop over the Claude API)
  that answers questions with a single Wikipedia tool and records its trajectory.
- **`eval/`** — an [Inspect AI](https://inspect.aisi.org.uk/) suite that scores
  the agent on benchmarks using an LLM-as-judge plus code-based scorers.

## Architecture & the one boundary

The subprojects are decoupled — separate folders, separate `uv` environments,
each runnable on its own. There is exactly **one** allowed coupling:

```
eval  ──imports──►  wiki_agent.run(question) -> AgentResult
```

Rules:
- **The agent must never import from `eval/`.** The dependency is one-way.
- Don't add new cross-project imports. If the eval needs more from the agent,
  extend `AgentResult` / the `run()` signature — keep the surface tiny.

File map:

| Path | Responsibility |
|------|----------------|
| `agent/wiki_agent/config.py` | Constants (model, endpoint, limits, backoff, cache) — single source of truth |
| `agent/wiki_agent/wikipedia.py` | The one tool: schema + actions + **pure parsers split from HTTP I/O**; `_get` adds maxlag, Retry-After/exponential backoff, and cache lookup |
| `agent/wiki_agent/cache.py` | Tiny isolated disk cache of raw API JSON (`agent/.wiki_cache/`, no eviction) |
| `agent/wiki_agent/ratelimit_bench.py` | Opt-in **live** benchmark: compares client setups for throttling, picks the best (pure scoring helpers tested) |
| `agent/wiki_agent/trajectory.py` | `Trajectory`/`Step` dataclasses + JSON persistence |
| `agent/wiki_agent/agent.py` | The loop: `run() -> AgentResult` |
| `agent/wiki_agent/cli.py` | `wiki-agent` CLI (`ask` default + `demo`) + `rich` trajectory rendering |
| `agent/wiki_agent/demos/` | Demo mode: `questions.py`, cached `*.json` trajectories, `player.py` (load/pick/play), `record.py` |
| `eval/wiki_eval/config.py` | Judge/agent model selection |
| `eval/wiki_eval/solver.py` | Wraps `wiki_agent.run` as an Inspect solver |
| `eval/wiki_eval/scorers.py` | LLM-judge + custom trajectory scorers |
| `eval/wiki_eval/tasks.py` | Benchmark registry (one `@task` each) |
| `eval/wiki_eval/datasets/*.jsonl` | Benchmark data (`{"input", "target"}`) |

## Development principles

- **Minimal and explicit** (Karpathy spirit). The manual tool-use loop is the
  point — don't replace it with a framework or the SDK tool-runner.
- **Separate pure logic from I/O.** Formatting/parsing must be testable without a
  network or API key (see `wikipedia._parse_*`). Keep new code the same way.
- **Small, single-purpose functions; clear names** (Clean Code).
- **Unit-test all custom logic** with fakes/monkeypatch — **no live API or
  network calls in tests.** Tests must pass with no `ANTHROPIC_API_KEY`.
- **Config lives in `config.py`.** Swapping Haiku → Sonnet is a one-line change
  there; don't hardcode model ids elsewhere.

## Commands

Run from within the relevant subproject (each has its own venv):

```bash
# agent/
uv sync
uv run pytest
uv run wiki-agent "Who was the first person to walk on the Moon?"
uv run wiki-agent ask "..." --no-cache       # bypass the Wikipedia disk cache
uv run wiki-agent ask "..." --clear-cache    # clear the cache, then run
uv run wiki-agent demo                 # replay a cached hard question (no API key)
uv run wiki-agent demo --record        # re-record cached demos (needs API key)
uv run python -m wiki_agent.ratelimit_bench  # live rate-limit comparison (opt-in, ~6 min)

# eval/
uv sync
uv run pytest
uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
uv run inspect view start --host 0.0.0.0 --port 7575   # results UI; 0.0.0.0 = reachable over Tailscale
```

## Models & environment

- Default model is **Haiku** (`claude-haiku-4-5`). Haiku does **not** support
  `thinking`/`effort` — don't add those params. The Sonnet upgrade path
  (`claude-sonnet-4-6`) is marked with a comment in `agent.py`.
- API key comes from `.env` (per subproject, **gitignored**) or the environment.
  `.env.example` documents the variable. Never commit a key.

## Wikipedia API etiquette

- **Anonymous "good citizenship"** is the chosen path — a compliant `User-Agent`
  (with contact + library/version) grants the 200 req/min tier; auth gives no
  read benefit under the 2026 rules. The agent is serial by construction.
- `_get` sends `maxlag=5`, retries 429/503 and in-200 `maxlag` bodies honoring
  `Retry-After` (else ≥5s then exponential). All knobs live in `config.py`.
- Responses are cached as raw JSON in `agent/.wiki_cache/` (gitignored, no
  eviction) keyed by the semantic params, so benchmark re-runs skip the network.
  `--no-cache` / `--clear-cache` (or `cache.clear()`) manage it.
- See `docs/superpowers/specs/2026-06-21-wikipedia-tool-ratelimit-cache-design.md`
  for the research and rationale.

## Git workflow — commit proactively (no need to ask)

When a unit of work is complete, **commit it** with a meaningful message (what
changed and why). You do not need to ask permission.

Multiple agents work in parallel on independent parts. **Never commit another
agent's work:**

- **Stage only the paths you changed**, e.g. `git add agent/wiki_agent/foo.py`.
  Do **not** use `git add -A` or `git add .` at the repo root.
- Run `git status` and review your staged diff (`git diff --cached`) before
  committing; if files you didn't touch are staged, unstage them.
- Keep each commit scoped to one subproject / one task.
- Do not create regular git branches, because that would mess with other agents working in the same repo. If needed create git worktrees or just work on master. 
- Never stage `.env` or other secrets (they're gitignored — keep it that way).

## Extending

- **New benchmark:** add `eval/wiki_eval/datasets/<name>.jsonl` + a `@task` in
  `tasks.py`.
- **New metric/scorer:** add a `@scorer` in `scorers.py` and include it in the
  task's scorer list.
- **Tool changes:** keep parsing pure and unit-tested; the tool returns readable
  strings and never raises into the model.
