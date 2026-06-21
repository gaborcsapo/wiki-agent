# Wikipedia Agent + Evaluation Suite

A small, from-scratch **agent** that answers questions using the Claude API and a
single Wikipedia tool, paired with an **evaluation suite** that scores it on four
benchmarks. The agent is a hand-written loop over the Claude API — no framework,
no SDK tool-runner — built to show the moving parts of an agentic system and how
to measure them.

```
$ wiki-agent "Which Apollo astronaut who walked on the Moon was born in Ohio?"
… searches Wikipedia, reads the relevant articles, reasons across them …
> Neil Armstrong — the first person to walk on the Moon — was born in
> Wapakoneta, Ohio.
```

Two deliberately **independent** subprojects:

| | What it is | Run on its own |
|---|---|---|
| [`agent/`](agent/README.md) | The agent: a loop over `messages.create` with one Wikipedia tool, plus a `rich` terminal CLI and a no-API-key demo mode. | ✅ |
| [`eval/`](eval/README.md) | An [Inspect AI](https://inspect.aisi.org.uk/) suite: LLM-as-judge + code-based scorers over four benchmarks. | ✅ |

## Quickstart

Both subprojects use [`uv`](https://docs.astral.sh/uv/) and each carries its own
environment. You need an Anthropic API key (the demo is the one exception).

```bash
# 1. Agent — answer a question live in the terminal
cd agent
uv sync
cp .env.example .env        # then paste your ANTHROPIC_API_KEY (or export it)
uv run wiki-agent "Who was the first person to walk on the Moon?"

# No API key handy? Replay a cached multi-hop trajectory:
uv run wiki-agent demo

# 2. Eval — score the agent on a benchmark
cd ../eval
uv sync
cp .env.example .env        # same key
uv run inspect eval wiki_eval/tasks.py@factual_qa --model anthropic/claude-haiku-4-5
uv run inspect view start   # browse results in the Inspect UI
```

The default agent model is **Haiku** (cheap and fast). For best quality, run with
Sonnet — see [Results](#results) below.

## How the agent works

The whole agent is a hand-written loop ([`agent/wiki_agent/agent.py`](agent/wiki_agent/agent.py)):

1. Send the question to Claude with **one tool available**: `wikipedia`.
2. If Claude wants to use the tool, execute the call against the MediaWiki API
   and feed the result back. Repeat.
3. When Claude stops requesting tools, that's the final answer.
4. If the step budget runs out first, make **one last call with no tools** —
   forcing the model to commit to its best answer instead of looping forever or
   returning a canned non-answer.

Every reasoning turn, tool call, and result is recorded as a **trajectory** —
rendered live in the CLI and consumed by the eval's scorers. The loop **never
raises into the model**: tool errors come back as readable strings, so a failed
fetch becomes information the model can react to, not a crash.

**The one tool** ([`agent/wiki_agent/wikipedia.py`](agent/wiki_agent/wikipedia.py))
exposes two actions — `search` (find articles) and `get_article` (read one's
plain-text body). It has three traits worth calling out:

- **Multilingual.** A `lang` parameter queries any Wikipedia edition (e.g.
  `lang='hu'`), since many facts live only on — or richer on — a native-language
  page.
- **Batchable.** `queries`/`titles` lists fan out independent lookups in parallel
  (bounded to 3 concurrent requests), cutting round-trips when work is
  parallelizable.
- **A good Wikipedia citizen.** A compliant `User-Agent` (unlocking the 200
  req/min tier), `maxlag=5`, and `Retry-After`/exponential backoff in `_get`,
  plus an on-disk cache of raw API JSON so benchmark re-runs skip the network.

See [`agent/README.md`](agent/README.md) for the CLI, demo mode, and file map.

## The benchmarks

Grading follows Anthropic's *Demystifying evals* guidance: a model-based
**LLM-as-judge** for open-ended correctness, plus fast **code-based** scorers
that inspect the trajectory. Every benchmark reports `accuracy` with `stderr`
error bars.

| Benchmark | Size | What it tests | Scorers |
|---|---:|---|---|
| **`factual_qa`** | 10 | Sanity check: straightforward single-fact questions. | correctness judge + tool-use check |
| **`frames`** | 100 | **Multi-hop reasoning** — questions from the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) benchmark that need 2–15 article reads and reasoning across them. | correctness judge + **retrieval grounding** (recall of gold pages) + tool-use |
| **`multilingual_qa`** | 30 | **Low-resource multilingual reach** — facts that live only on (or richer on) a non-English Wikipedia, and questions *asked* in another language. | language-neutral judge with per-category / per-language breakdowns |
| **`abstention`** | 36 | **Knowing when not to answer** — does the agent decline/flag instead of hallucinating? 30 unanswerable questions (false premise, unknowable, stale, underspecified, subjective, garbled, out-of-scope) + 6 answerable controls. Inspired by Meta FAIR's [AbstentionBench](https://arxiv.org/abs/2506.09038). | binary **abstention judge** → precision / recall / F1 |

Each probes a different competence: *can it find a fact* (`factual_qa`), *can it
chain facts* (`frames`), *can it reach beyond English* (`multilingual_qa`), and
*can it stay quiet when it should* (`abstention`).

## Results

Full reruns of each benchmark on both the **Haiku** (shipped default) and
**Sonnet 4.6** (quality tier) agent. The LLM-judge is held at Haiku throughout,
so the deltas reflect the *agent*, not the grader. Every run completed all
samples cleanly (no errors). Reproduce any row with the canonical runner —
`cd eval && uv run python run_pinned.py <task> <agent-model> <limit>` (e.g.
`run_pinned.py frames claude-sonnet-4-6 100`).

**`frames` — multi-hop reasoning** (100 questions)

| Metric | Haiku | Sonnet |
|---|---:|---:|
| **Correctness** | 0.57 | **0.76** |
| Grounding recall (gold pages read) | 0.52 | 0.54 |
| Avg. steps / question | 14.7 | 10.8 |

Sonnet is both **more accurate and more efficient** — it reaches the answer in
fewer steps (10.8 vs 14.7) because it picks the right page sooner, where Haiku
flails across more reads. (±0.05 stderr.)

**`multilingual_qa` — low-resource reach** (30 questions)

| Metric | Haiku | Sonnet |
|---|---:|---:|
| **Correctness** | 0.30 | **0.53** |
| — facts only on a non-English page (`cross_lingual_fact`) | 0.08 | 0.50 |
| — obscure people, richer native page (`richer_native_page`) | 0.30 | 0.40 |
| — question asked in another language (`foreign_language_query`) | 0.63 | 0.75 |

The `lang` tool support gets the agent *to* the native page; comprehending it is
the bottleneck — and that's where Sonnet pulls far ahead (cross-lingual facts
0.08 → 0.50). (±0.09 stderr; n=30.)

**`abstention` — knowing when not to answer** (36 questions: 30 unanswerable + 6 controls)

| Metric | Haiku | Sonnet |
|---|---:|---:|
| Accuracy | **0.89** | 0.81 |
| Abstention recall (caught the unanswerable) | 0.87 | 0.77 |
| Abstention precision (didn't over-abstain) | 1.00 | 1.00 |
| **F1** | **0.93** | 0.87 |

The one benchmark where **Haiku wins**: both models have perfect precision (never
wrongly abstained on the 6 answerable controls), but Sonnet's confidence makes it
*answer* more of the unanswerable questions, so it catches fewer (recall 0.77 vs
0.87). A stronger model is not automatically a more cautious one. (±0.06 stderr; n=36.)

## Where the agent still fails

- **Numeric & temporal multi-hop (FRAMES).** The residual ~24% of FRAMES misses
  are dominated by *arithmetic over correctly-retrieved facts* — "how many years
  after the city's founding…", "the building's height in feet equals the
  number…", population-as-of-a-date comparisons. The agent finds the facts but
  slips on the composition. A scratchpad/compute step is the open lever.
- **Reading a foreign page, not just reaching it (multilingual).** With `lang`,
  the agent reliably opens the native edition (tool-use is 100%), but on Haiku
  the fact in front of it doesn't survive translation/extraction —
  `cross_lingual_fact` sits at 0.08. This is a *comprehension* ceiling, not a
  retrieval one: Sonnet lifts it to 0.50 on the same tool.
- **False premises & the genuinely unknowable (abstention).** Sonnet's misses
  cluster in two categories: `false_premise` ("In what year was the Great Wall of
  China torn down?") and `unknowable` ("What did Julius Caesar dream the night
  before he was assassinated?"). It searches, finds related material, and answers
  *around* the false premise instead of flagging it.
- **Cost of the weak model's flailing.** Haiku spends ~14.7 steps and ~2.4× the
  tokens of Sonnet on FRAMES for a lower score — when a question is hard, the
  cheaper model doesn't fail fast, it loops.

## Hill-climbing: how the scores got here

The agent wasn't tuned by guessing — it was hill-climbed against the benchmarks,
one change at a time, with the judge held constant so each delta measures the
*agent*, not the grader. The full log lives in
[`docs/hill-climbing-report.md`](docs/hill-climbing-report.md); the headline
findings:

| Change | Impact | Lesson |
|---|---|---|
| Haiku → **Sonnet 4.6** | **+0.20** correctness (FRAMES) | Model tier is the single biggest lever. |
| **Decisive system prompt** (decompose, commit, don't loop) | +0.15 | The baseline agent wandered and ran out of steps; telling it to commit fixed more than more steps did. |
| Fetch article **body**, not just the intro | +0.10 | Facts sit below the lead — intro-only retrieval found the right page but not the fact. |
| **`lang` support** (query native editions) | +0.20 on `multilingual_qa` | English-only retrieval simply can't reach facts that only exist on other editions. |
| "Turn the knobs up" accuracy levers | **+0.10 Haiku / −0.05 Sonnet** | The same levers help a weak model and *distract* a strong one — tune to the tier. |

Things that were tried and **dropped**: extended/adaptive thinking, chain-of-thought
prompting, and a structured output format (no gain, more latency); parallel
multi-query lookups (kept but unused — FRAMES hops are sequentially dependent,
so there's nothing to parallelize).

## How this project was built

Roughly in the order it happened:

1. **The tool first, hardened.** Before any tuning, the Wikipedia tool was made a
   good API citizen: an empirical [rate-limit benchmark](agent/wiki_agent/ratelimit_bench.py)
   established that a compliant `User-Agent` unlocks the 200 req/min tier, and
   `maxlag` + `Retry-After`/backoff + a disk cache made dozens of benchmark
   re-runs cheap and well-behaved.
2. **The loop and the prompt.** A baseline loop revealed the agent was running out
   of steps and giving up. A decisive prompt rewrite (+0.15) and fetching the
   article body (+0.10) did most of the early work.
3. **The model axis.** Swapping Haiku → Sonnet was the biggest single win
   (+0.20). Extended thinking and chain-of-thought were tested here and dropped —
   the multi-step tool loop *is* the reasoning scaffold, so bolting on more
   reasoning just added latency.
4. **Parallel lookups.** Batched `queries`/`titles` were added so the model could
   fan out independent lookups. It works, but FRAMES barely uses it (its hops are
   sequential) — kept because it's free when idle and pays off elsewhere.
5. **Multilingual.** A `lang` parameter let the agent query native editions —
   a 2.5× win on multilingual questions, with a new benchmark and analyzer to
   measure it.
6. **Abstention.** A benchmark and binary judge for the harder skill of declining
   to answer the unanswerable, with answerable controls to catch over-abstention.
7. **Demo mode.** An `on_step` callback drives both live CLI rendering and
   replay of cached trajectories, so the agent can be shown off without an API key.

Each feature was **spec'd, then implemented, then measured** — the specs and
plans live under [`docs/superpowers/`](docs/superpowers/), and the tuning history
in [`docs/hill-climbing-report.md`](docs/hill-climbing-report.md).

## Repo map

```
agent/                  the agent (own venv, own README)
  wiki_agent/
    agent.py            the loop: run() -> AgentResult
    wikipedia.py        the one tool (pure parsers split from HTTP I/O)
    cache.py            disk cache of raw API JSON
    config.py           all tunable constants (model, limits, backoff)
    trajectory.py       trajectory/step dataclasses + JSON persistence
    cli.py              the wiki-agent CLI + rich rendering
    demos/              cached trajectories + replay player
    ratelimit_bench.py  opt-in live rate-limit benchmark
eval/                   the Inspect suite (own venv, own README)
  wiki_eval/
    solver.py           wraps wiki_agent.run as an Inspect solver
    scorers.py          LLM-judge + code-based trajectory scorers
    tasks.py            one @task per benchmark
    datasets/           the four benchmark datasets (JSONL)
    analyze_*.py        post-run breakdowns (FRAMES, multilingual)
docs/
  hill-climbing-report.md     the tuning log (read this for the "why")
  superpowers/                per-feature specs, plans, and reports
```

## Tests

```bash
cd agent && uv run pytest    # tool parsers, trajectory, loop (fake client)
cd eval  && uv run pytest    # solver adapter, scorers (fakes, no API key)
```

Both suites run offline with no API key.
