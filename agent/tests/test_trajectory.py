"""Unit tests for trajectory recording and serialization."""

import json

from wiki_agent.trajectory import FINAL_ANSWER, TOOL_CALL, Step, Trajectory


def test_to_dict_round_trips_through_json():
    traj = Trajectory(question="Q?", model="claude-haiku-4-5")
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search", "query": "q"}))
    traj.add(Step(kind=FINAL_ANSWER, content="A."))
    traj.answer = "A."

    restored = json.loads(json.dumps(traj.to_dict()))
    assert restored["question"] == "Q?"
    assert restored["answer"] == "A."
    assert len(restored["steps"]) == 2
    assert restored["steps"][0]["tool_input"] == {"action": "search", "query": "q"}


def test_save_writes_json_file(tmp_path):
    traj = Trajectory(question="Q?", model="m")
    traj.add(Step(kind=FINAL_ANSWER, content="hi"))
    path = traj.save(tmp_path)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["question"] == "Q?"
    assert data["steps"][0]["content"] == "hi"
