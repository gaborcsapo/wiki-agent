"""CLI routing tests using click's CliRunner — no network, no real sleep."""

from click.testing import CliRunner

from wiki_agent import cli
from wiki_agent.agent import AgentResult
from wiki_agent.trajectory import FINAL_ANSWER, Step, Trajectory


def _traj(question="Q?"):
    t = Trajectory(question=question, model="m")
    t.add(Step(kind=FINAL_ANSWER, content="the answer"))
    t.answer = "the answer"
    return t


def test_bare_question_routes_to_ask(monkeypatch):
    captured = {}

    def fake_run(question, *, model=None, max_steps=6, on_step=None):
        captured["question"] = question
        # Drive the live renderer the way the real run would.
        for step in _traj(question).steps:
            if on_step:
                on_step(step)
        return AgentResult(answer="the answer", trajectory=_traj(question), steps=1)

    monkeypatch.setattr(cli, "run", fake_run)
    result = CliRunner().invoke(cli.cli, ["Who walked on the moon?", "--no-save"])

    assert result.exit_code == 0, result.output
    assert captured["question"] == "Who walked on the moon?"
    assert "the answer" in result.output


def test_demo_plays_random_cached_trajectory(monkeypatch):
    monkeypatch.setattr(cli, "load_demos", lambda: [_traj("Demo Q?")])
    monkeypatch.setattr(cli, "pick_demo", lambda trajs: trajs[0])
    # Make playback instant by stubbing the sleep used inside play().
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)

    result = CliRunner().invoke(cli.cli, ["demo"])

    assert result.exit_code == 0, result.output
    assert "Demo Q?" in result.output
    assert "the answer" in result.output
