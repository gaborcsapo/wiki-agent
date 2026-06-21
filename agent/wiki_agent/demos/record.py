"""Record the curated demo trajectories.

This is the only demo component that needs an ``ANTHROPIC_API_KEY`` and network
access: it runs each question through the real agent and saves the resulting
trajectory JSON. The CLI exposes it as ``wiki-agent demo --record``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import run
from .player import DEMOS_DIR
from .questions import QUESTIONS


def record_demos(directory: Path = DEMOS_DIR, *, questions=QUESTIONS, run_fn=run) -> list[Path]:
    """Run each question through ``run_fn`` and write ``NN.json`` files.

    Returns the list of written paths. Stable, zero-padded filenames mean a
    re-record overwrites the previous set in place.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, question in enumerate(questions):
        result = run_fn(question)
        path = directory / f"{index:02d}.json"
        path.write_text(
            json.dumps(result.trajectory.to_dict(), indent=2, ensure_ascii=False)
        )
        paths.append(path)
    return paths
