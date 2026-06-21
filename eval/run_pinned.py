"""Run an Inspect task with the agent model pinned IN THIS PROCESS.

The shared default `wiki_agent.config.AGENT_MODEL` can be edited on disk by other
work in parallel; pinning it in memory here makes an experiment run reproducible
regardless of those edits.

    uv run python run_pinned.py <task> [model] [limit]
    uv run python run_pinned.py multilingual_qa claude-haiku-4-5 30
"""

import sys

import wiki_agent.config as agent_config


def main() -> None:
    from wiki_eval import config as eval_config

    task_name = sys.argv[1] if len(sys.argv) > 1 else "frames"
    model = sys.argv[2] if len(sys.argv) > 2 else agent_config.AGENT_MODEL
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    agent_config.AGENT_MODEL = model  # pin in-process (immune to on-disk edits)

    from inspect_ai import eval as inspect_eval
    from wiki_eval import tasks

    task = getattr(tasks, task_name)()
    # The judge model is the single source of truth in wiki_eval/config.py — don't
    # hardcode it here, or a Haiku->Sonnet judge swap would silently not apply.
    logs = inspect_eval(task, model=eval_config.JUDGE_MODEL, limit=limit)
    print("PINNED_AGENT_MODEL:", model)
    print("LOG:", logs[0].location)


if __name__ == "__main__":
    main()
