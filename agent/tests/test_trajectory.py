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


def test_from_dict_round_trips_to_dict():
    from wiki_agent.trajectory import ASSISTANT_TEXT, TOOL_RESULT

    traj = Trajectory(question="Q?", model="claude-haiku-4-5")
    traj.add(Step(kind=ASSISTANT_TEXT, content="thinking", input_tokens=10, output_tokens=5))
    traj.add(Step(kind=TOOL_CALL, tool_name="wikipedia", tool_input={"action": "search", "query": "q"}))
    traj.add(Step(kind=TOOL_RESULT, content="result text", tool_name="wikipedia"))
    traj.add(Step(kind=FINAL_ANSWER, content="A."))
    traj.answer = "A."

    restored = Trajectory.from_dict(traj.to_dict())

    assert restored == traj
    assert restored.steps[1].tool_input == {"action": "search", "query": "q"}
    assert restored.steps[0].output_tokens == 5


def test_from_dict_ignores_unknown_step_keys():
    # A demo JSON written by a different Step schema must still replay.
    data = {
        "question": "Q?",
        "model": "m",
        "answer": "A.",
        "steps": [{"kind": "assistant_text", "content": "hi", "future_field": 123}],
    }
    traj = Trajectory.from_dict(data)
    assert traj.steps[0].content == "hi"
    assert traj.answer == "A."


def test_save_writes_json_file(tmp_path):
    traj = Trajectory(question="Q?", model="m")
    traj.add(Step(kind=FINAL_ANSWER, content="hi"))
    path = traj.save(tmp_path)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["question"] == "Q?"
    assert data["steps"][0]["content"] == "hi"
