"""Load cached demo trajectories and play them back step by step.

Pure of any rendering or API concerns: ``play`` takes the renderer and the
sleep function as parameters, so the CLI supplies the real ones and tests
supply fakes (no network, no real sleeping).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

from ..trajectory import FINAL_ANSWER, Step, Trajectory

# Cached trajectory JSONs live alongside this module.
DEMOS_DIR = Path(__file__).resolve().parent

# Fixed pause before revealing each step, to simulate a live run.
STEP_DELAY_SECONDS = 0.8


def load_demos(directory: Path = DEMOS_DIR) -> list[Trajectory]:
    """Parse every ``*.json`` demo in ``directory``, sorted by filename."""
    return [
        Trajectory.from_dict(json.loads(path.read_text()))
        for path in sorted(directory.glob("*.json"))
    ]


def pick_demo(trajectories: list[Trajectory], rng=random) -> Trajectory:
    """Pick one trajectory at random (``rng`` injectable for tests)."""
    return rng.choice(trajectories)


def play(
    traj: Trajectory,
    render_step: Callable[[Step], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    delay: float = STEP_DELAY_SECONDS,
) -> None:
    """Reveal each non-final step through ``render_step`` with a pause before each."""
    for step in traj.steps:
        if step.kind == FINAL_ANSWER:
            continue  # the final answer panel is printed by the caller
        sleep(delay)
        render_step(step)
