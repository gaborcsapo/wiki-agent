# wiki-eval

An [Inspect AI](https://inspect.aisi.org.uk/) evaluation suite for the Wikipedia
agent. It measures how well the agent answers questions across several
benchmarks — factual QA, multi-hop FRAMES, low-resource multilingual QA, and
abstention — graded by an LLM-as-judge plus code-based trajectory scorers.

## Decoupling

This suite is independent of the agent except for **one import**: the solver
calls `wiki_agent.run(question)`. The agent is declared as an editable path
dependency (`../agent`) and has zero knowledge of this suite. You can run the
agent without the eval, and the eval depends on the agent only through that
single function.

## Taxonomy (Inspect ↔ Anthropic "Demystifying evals")

| Inspect term | Meaning here |
|--------------|--------------|
| **Task** | A benchmark: dataset + solver + scorers (`tasks.py`) |
| **Sample** | One test case: a question (`input`) + grading rubric (`target`) |
| **Solver** | The thing under test — our agent (`solver.py`) |
| **Scorer** | The grader. We use model-based (LLM-judge) + a code-based check (`scorers.py`) |
| **Transcript** | Per-sample trace (incl. the agent's trajectory) in `inspect view` |
| **Epochs** | Repeat-sampling per question (for future statistical rigor) |

Grading mirrors Anthropic's guidance: a model-based **LLM-as-judge** for
open-ended correctness, plus a fast **code-based** trajectory check that the
agent actually used its tool. Metrics report `accuracy()` with `stderr()` (error
bars).

## Layout

| File | Responsibility |
|------|----------------|
| `wiki_eval/config.py` | Judge/agent model selection (one-line Haiku → Sonnet swap) |
| `wiki_eval/solver.py` | Wraps `wiki_agent.run` as an Inspect solver |
| `wiki_eval/scorers.py` | LLM-judge correctness + custom trajectory scorer |
| `wiki_eval/tasks.py` | Benchmark registry (one `@task` per benchmark) |
| `wiki_eval/datasets/factual_qa.jsonl` | 10 hand-written QA examples |
| `wiki_eval/datasets/frames.jsonl` | 100 multi-hop FRAMES questions with `reference_pages` |
| `wiki_eval/datasets/multilingual_qa.jsonl` | 30 low-resource multilingual QA questions |
| `wiki_eval/datasets/abstention.jsonl` | 30 should-abstain questions + 6 answerable controls |

## Setup

```bash
cd eval
uv sync
```

Provide your Anthropic API key either by editing `eval/.env` (gitignored —
Inspect AI auto-loads it) or by exporting it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run an eval

```bash
uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
```

The agent calls Claude directly, so it never goes through Inspect's `--model`;
that flag only sets the run's *label* and the judge's fallback. The agent runs at
`config.AGENT_MODEL` (default Haiku), and every scorer pins the judge to
`JUDGE_MODEL` explicitly.

### Running a specific agent tier (recommended)

To run — and correctly log — the agent at a chosen tier, use `run_pinned.py`. It
pins the agent model in-process *and* labels the run with it, so the Inspect UI's
**Model** and **Tokens** columns reflect the agent (not the judge):

```bash
uv run python run_pinned.py <task> <agent-model> <limit>
uv run python run_pinned.py frames claude-sonnet-4-6 100
uv run python run_pinned.py multilingual_qa claude-haiku-4-5 30
```

To use a stronger judge instead, change `JUDGE_MODEL` in `wiki_eval/config.py`.

## Benchmarks

### factual_qa

10 hand-written factual questions graded by the LLM-judge `correctness_judge`,
plus the `used_wikipedia_tool` trajectory check.

```bash
uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
```

### frames

100 multi-hop questions from the FRAMES benchmark. Each row carries
`reference_pages` (the English Wikipedia articles needed to answer), scored by
`correctness_judge` + `used_wikipedia_tool` + `retrieval_grounding` (recall of
the read pages vs. the reference set). Use `scripts/build_frames.py` to
(re)build the dataset and `python -m wiki_eval.analyze_runs <log.eval>` for a
post-run breakdown.

```bash
uv run inspect eval wiki_eval/tasks.py@frames --model anthropic/claude-haiku-4-5
```

### multilingual_qa

30 low-resource multilingual QA questions whose facts are often only on (or
richer on) a non-English Wikipedia edition, exercising the tool's `lang`
parameter. Scored by `multilingual_correctness` (with per-category/language
metrics) + `used_wikipedia_tool`; `python -m wiki_eval.analyze_multilingual
<log.eval>` lists failures and the language editions the agent queried.

```bash
uv run inspect eval wiki_eval/tasks.py@multilingual_qa --model anthropic/claude-haiku-4-5
```

### WikiAgentAbstention

Measures whether the agent **abstains** — declines, asks to clarify, or flags a
problem — instead of fabricating an answer it cannot ground. 30
abstention-positive questions across seven categories (false premise,
unknowable, stale/real-time, underspecified, subjective, garbled voice-typing,
out-of-scope) plus 6 answerable controls to catch over-abstention. Inspired by
Meta FAIR's AbstentionBench (arXiv:2506.09038).

A binary `abstention_judge` labels each answer ABSTAIN/ANSWER (a confident answer
with a token caveat still counts as ANSWER) and grades it against the row's
`should_abstain`. Metrics: abstention **recall** (caught the unanswerable),
**precision** (didn't over-abstain on controls), **F1**, and overall accuracy.
`used_wikipedia_tool` is reported as a diagnostic only — for garbled/out-of-scope
rows, *not* searching is often the correct behavior.

```bash
uv run inspect eval wiki_eval/tasks.py@wiki_agent_abstention --model anthropic/claude-haiku-4-5
```

## View results (over Tailscale)

```bash
uv run inspect view start --host 0.0.0.0 --port 7575
```

Binding `0.0.0.0` exposes the viewer beyond localhost so you can open it from
your laptop via the host's Tailscale address/MagicDNS name
(`http://<tailscale-host>:7575`). The default `127.0.0.1` is localhost-only.

## Add a benchmark

1. Add `wiki_eval/datasets/<name>.jsonl` (`{"input": ..., "target": ...}` rows).
2. Add a `@task` in `tasks.py` pointing at it, with whatever scorers fit.

## Test

```bash
uv run pytest
```

Tests cover the solver adapter and the custom scorer with fakes — no API key
required.
