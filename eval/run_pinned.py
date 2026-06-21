"""Run an Inspect task with the agent model pinned IN THIS PROCESS.

This is the canonical way to run an eval at a chosen agent tier: it pins the
agent model in memory (immune to on-disk edits by parallel work) AND labels the
Inspect run with that model, so `inspect view` shows the correct Model and Tokens
for the agent — not the judge. (The plain `inspect eval ... --model X` CLI sets
the run's *label* to X but always runs the agent at `config.AGENT_MODEL`, so the
two can disagree; prefer this runner when the agent tier matters.)

    uv run python run_pinned.py <task> [model] [limit]
    uv run python run_pinned.py multilingual_qa claude-haiku-4-5 30
"""

import sys

import wiki_agent.config as agent_config


def main() -> None:
    from wiki_eval import config as eval_config  # noqa: F401  (judge model lives here)

    task_name = sys.argv[1] if len(sys.argv) > 1 else "frames"
    model = sys.argv[2] if len(sys.argv) > 2 else agent_config.AGENT_MODEL
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    agent_config.AGENT_MODEL = model  # pin in-process (immune to on-disk edits)

    from inspect_ai import eval as inspect_eval
    from wiki_eval import tasks

    task = getattr(tasks, task_name)()
    # Label the run with the AGENT model (provider-qualified) so the Inspect UI's
    # Model/Tokens reflect the agent. The judge is independent — every scorer
    # selects wiki_eval.config.JUDGE_MODEL explicitly, so it is unaffected by the
    # model passed here.
    logs = inspect_eval(task, model=f"anthropic/{model}", limit=limit)
    print("PINNED_AGENT_MODEL:", model)
    print("LOG:", logs[0].location)


if __name__ == "__main__":
    main()
