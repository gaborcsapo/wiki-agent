from wiki_eval.render import MAX_RESULT_CHARS, format_trajectory


def _traj(steps, answer="42"):
    return {"question": "Q?", "model": "claude-haiku-4-5", "answer": answer, "steps": steps}


def test_renders_outputs_and_tools():
    out = format_trajectory(_traj(
        steps=[
            {"kind": "assistant_text", "content": "Let me search."},
            {"kind": "tool_call", "tool_name": "get_article", "tool_input": {"title": "Moon"}},
            {"kind": "tool_result", "content": "The Moon is a satellite."},
        ],
        answer="The answer is 42.",
    ))
    # question, model, assistant text, tool name + args, result, answer all present
    for expected in ("Q?", "claude-haiku-4-5", "Let me search.",
                     "get_article", "Moon", "The Moon is a satellite.", "The answer is 42."):
        assert expected in out


def test_truncates_long_tool_result():
    big = "x" * (MAX_RESULT_CHARS + 500)
    out = format_trajectory(_traj([{"kind": "tool_result", "content": big}]))
    assert "truncated" in out
    assert out.count("x") <= MAX_RESULT_CHARS


def test_robust_to_unknown_kind_and_missing_fields():
    # unknown step kind and a tool_call missing tool_name/tool_input must not raise
    out = format_trajectory({"steps": [{"kind": "mystery", "content": "??"}, {"kind": "tool_call"}]})
    assert isinstance(out, str)
