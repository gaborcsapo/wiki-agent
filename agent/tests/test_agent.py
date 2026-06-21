"""Unit tests for the agent loop, using a scripted fake Claude client."""

from types import SimpleNamespace

import pytest

from wiki_agent import agent, wikipedia
from wiki_agent.trajectory import FINAL_ANSWER, TOOL_CALL, TOOL_RESULT


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(name, input_, id_="tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _response(content, stop_reason):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class FakeClient:
    """Returns a scripted list of responses, one per messages.create call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _stub_wikipedia(monkeypatch):
    monkeypatch.setattr(wikipedia, "dispatch", lambda inp, **kw: f"RESULT:{inp['action']}")


def test_direct_answer_without_tool_use():
    client = FakeClient([_response([_text_block("Paris.")], "end_turn")])
    result = agent.run("Capital of France?", client=client)

    assert result.answer == "Paris."
    assert result.steps == 1
    assert result.trajectory.steps[-1].kind == FINAL_ANSWER
    assert len(client.calls) == 1


def test_tool_use_then_answer():
    client = FakeClient(
        [
            _response([_tool_block("wikipedia", {"action": "search", "query": "moon"})], "tool_use"),
            _response([_text_block("Neil Armstrong.")], "end_turn"),
        ]
    )
    result = agent.run("First on the moon?", client=client)

    assert result.answer == "Neil Armstrong."
    assert result.steps == 2
    kinds = [s.kind for s in result.trajectory.steps]
    assert TOOL_CALL in kinds and TOOL_RESULT in kinds and FINAL_ANSWER in kinds
    # Second API call must include the tool result in history.
    assert len(client.calls) == 2


def test_on_step_callback_fires_per_recorded_step():
    client = FakeClient(
        [
            _response([_tool_block("wikipedia", {"action": "search", "query": "moon"})], "tool_use"),
            _response([_text_block("Neil Armstrong.")], "end_turn"),
        ]
    )
    seen = []
    result = agent.run("Who walked on the moon?", client=client, on_step=seen.append)

    # Same steps the trajectory recorded, in the same order.
    assert [s.kind for s in seen] == [s.kind for s in result.trajectory.steps]
    assert seen[-1].kind == FINAL_ANSWER
    # The objects handed to the callback ARE the recorded steps.
    assert seen == result.trajectory.steps


def test_respects_max_steps():
    # Always asks for a tool — should stop at the cap, not loop forever. After the
    # cap, one final tool-less call forces a best-effort answer.
    looping = [
        _response([_tool_block("wikipedia", {"action": "search", "query": "x"})], "tool_use")
        for _ in range(3)
    ]
    forced = _response([_text_block("Best effort answer.")], "end_turn")
    client = FakeClient(looping + [forced])
    result = agent.run("never ends", client=client, max_steps=3)

    assert result.steps == 3
    assert len(client.calls) == 4  # 3 tool-loop steps + 1 forced final answer
    assert result.answer == "Best effort answer."
