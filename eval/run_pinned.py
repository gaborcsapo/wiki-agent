"""Run the FRAMES task with the agent model pinned IN THIS PROCESS.

The shared default `wiki_agent.config.AGENT_MODEL` can be edited on disk by other
work in parallel; pinning it in memory here makes an experiment run reproducible
regardless of those edits. Used for the parallel-multiquery FRAMES experiment.

    uv run python run_pinned.py [model] [limit]
"""

import sys

import wiki_agent.config as agent_config


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-4-5"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    agent_config.AGENT_MODEL = model  # pin in-process (immune to on-disk edits)

    from inspect_ai import eval as inspect_eval
    from wiki_eval.tasks import frames

    logs = inspect_eval(frames(), model="anthropic/claude-haiku-4-5", limit=limit)
    print("PINNED_AGENT_MODEL:", model)
    print("LOG:", logs[0].location)


if __name__ == "__main__":
    main()
