"""Unit tests for demo loading and playback — no network, no real sleep."""

import json

from wiki_agent.demos import player
from wiki_agent.trajectory import (
    ASSISTANT_TEXT,
    FINAL_ANSWER,
    TOOL_CALL,
    Step,
    Trajectory,
)


def _sample_trajectory(question: str) -> Trajectory:
    traj = Trajectory(question=question, model="claude-haiku-4-5")
    traj.add(Step(kind=ASSISTANT_TEXT, content="let me look this up"))
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search", "query": "x"}))
    traj.add(Step(kind=FINAL_ANSWER, content="the answer"))
    traj.answer = "the answer"
    return traj


def _write_demo(directory, name, traj):
    (directory / name).write_text(json.dumps(traj.to_dict(), ensure_ascii=False))


def test_load_demos_parses_all_json_sorted(tmp_path):
    _write_demo(tmp_path, "01.json", _sample_trajectory("Q1?"))
    _write_demo(tmp_path, "00.json", _sample_trajectory("Q0?"))

    loaded = player.load_demos(tmp_path)

    assert [t.question for t in loaded] == ["Q0?", "Q1?"]
    assert isinstance(loaded[0], Trajectory)


def test_load_demos_skips_corrupt_file(tmp_path):
    _write_demo(tmp_path, "00.json", _sample_trajectory("Good?"))
    (tmp_path / "01.json").write_text("{ this is not valid json")

    loaded = player.load_demos(tmp_path)

    # The corrupt file is skipped, not fatal.
    assert [t.question for t in loaded] == ["Good?"]


def test_pick_demo_uses_injected_rng():
    trajs = [_sample_trajectory("A?"), _sample_trajectory("B?")]

    class FakeRng:
        def choice(self, seq):
            return seq[-1]

    assert player.pick_demo(trajs, rng=FakeRng()).question == "B?"


def test_record_demos_writes_one_json_per_question(tmp_path):
    from wiki_agent.demos import record
    from wiki_agent.agent import AgentResult

    def fake_run(question):
        return AgentResult(answer="A.", trajectory=_sample_trajectory(question), steps=1)

    paths = record.record_demos(
        tmp_path, questions=["Q-one?", "Q-two?"], run_fn=fake_run
    )

    assert [p.name for p in paths] == ["00.json", "01.json"]
    data = json.loads((tmp_path / "00.json").read_text())
    assert data["question"] == "Q-one?"
    assert data["steps"][0]["kind"] == "assistant_text"


def test_record_demos_prunes_stale_files_from_larger_previous_set(tmp_path):
    from wiki_agent.demos import record
    from wiki_agent.agent import AgentResult

    def fake_run(question):
        return AgentResult(answer="A.", trajectory=_sample_trajectory(question), steps=1)

    # A previous, larger set leaves 02.json behind.
    (tmp_path / "02.json").write_text("stale")
    paths = record.record_demos(tmp_path, questions=["Q-one?", "Q-two?"], run_fn=fake_run)

    assert [p.name for p in paths] == ["00.json", "01.json"]
    assert not (tmp_path / "02.json").exists()  # stale demo pruned


def test_play_renders_non_final_steps_with_injected_sleep():
    traj = _sample_trajectory("Q?")
    rendered = []
    slept = []

    player.play(traj, rendered.append, sleep=slept.append, delay=0.5)

    # FINAL_ANSWER is skipped; the two preceding steps are rendered in order.
    assert [s.kind for s in rendered] == [ASSISTANT_TEXT, TOOL_CALL]
    # One sleep per rendered step, using the supplied delay; no real sleeping.
    assert slept == [0.5, 0.5]
