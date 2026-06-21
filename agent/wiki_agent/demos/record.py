"""Record the curated demo trajectories.

This is the only demo component that needs an ``ANTHROPIC_API_KEY`` and network
access: it runs each question through the real agent and saves the resulting
trajectory JSON. The CLI exposes it as ``wiki-agent demo --record``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..agent import run
from .player import DEMOS_DIR
from .questions import QUESTIONS


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + rename, so an interrupted
    re-record never leaves a half-written demo for the player to choke on."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def record_demos(directory: Path = DEMOS_DIR, *, questions=QUESTIONS, run_fn=run) -> list[Path]:
    """Run each question through ``run_fn`` and write ``NN.json`` files.

    Returns the list of written paths. Each file is written atomically, and any
    stale ``NN.json`` left over from a previous, larger set is pruned so the demo
    set always matches ``questions`` exactly.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, question in enumerate(questions):
        result = run_fn(question)
        path = directory / f"{index:02d}.json"
        _atomic_write_text(
            path, json.dumps(result.trajectory.to_dict(), indent=2, ensure_ascii=False)
        )
        paths.append(path)
    for stale in directory.glob("[0-9][0-9].json"):
        if stale not in paths:
            stale.unlink()
    return paths
