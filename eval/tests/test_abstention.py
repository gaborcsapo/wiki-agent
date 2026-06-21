"""Unit tests for the abstention benchmark scorer (no API calls)."""

import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, SampleScore, Score

from wiki_eval.scorers import (
    _abstention_prompt,
    _parse_abstention_verdict,
    _prf,
    abstention_f1,
    abstention_judge,
    abstention_precision,
    abstention_recall,
)


# --- _prf (confusion-matrix math) ----------------------------------------


def test_prf_all_correct():
    # 2 abstain-positives both caught, 1 control not abstained on.
    pairs = [(True, True), (True, True), (False, False)]
    out = _prf(pairs)
    assert out == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_prf_under_abstention_lowers_recall():
    # 2 should-abstain, only 1 caught -> recall 0.5, precision 1.0.
    pairs = [(True, True), (True, False), (False, False)]
    out = _prf(pairs)
    assert out["recall"] == 0.5
    assert out["precision"] == 1.0
    assert out["f1"] == pytest.approx(2 / 3)


def test_prf_over_abstention_lowers_precision():
    # 1 should-abstain caught, but also abstained on a control -> precision 0.5.
    pairs = [(True, True), (False, True)]
    out = _prf(pairs)
    assert out["recall"] == 1.0
    assert out["precision"] == 0.5


def test_prf_zero_denominators():
    assert _prf([]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # No predicted positives -> precision 0; no actual positives -> recall 0.
    assert _prf([(False, False)]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# --- metrics over SampleScores -------------------------------------------


def _sample(should_abstain: bool, did_abstain: bool) -> SampleScore:
    value = CORRECT if should_abstain == did_abstain else INCORRECT
    return SampleScore(
        score=Score(
            value=value,
            metadata={"should_abstain": should_abstain, "did_abstain": did_abstain},
        )
    )


def test_metrics_read_sample_metadata():
    scores = [
        _sample(True, True),    # TP
        _sample(True, False),   # FN (under-abstention)
        _sample(False, True),   # FP (over-abstention)
        _sample(False, False),  # TN
    ]
    assert abstention_recall()(scores) == 0.5      # 1 of 2 should-abstain caught
    assert abstention_precision()(scores) == 0.5   # 1 of 2 abstentions appropriate
    assert abstention_f1()(scores) == pytest.approx(0.5)


# --- pure judge helpers --------------------------------------------------


def test_prompt_contains_question_answer_and_caveat_rule():
    p = _abstention_prompt("How tall is the tower?", "It is 300 m.")
    assert "How tall is the tower?" in p
    assert "It is 300 m." in p
    assert "caveat" in p.lower()
    assert "VERDICT: ABSTAIN" in p and "VERDICT: ANSWER" in p


def test_parse_verdict_abstain_and_answer():
    assert _parse_abstention_verdict("Reasoning...\nVERDICT: ABSTAIN") is True
    assert _parse_abstention_verdict("VERDICT: ANSWER") is False
    assert _parse_abstention_verdict("verdict: abstain") is True


def test_parse_verdict_uses_last_token():
    text = "VERDICT: ANSWER\nactually no\nVERDICT: ABSTAIN"
    assert _parse_abstention_verdict(text) is True


def test_parse_verdict_unparseable_defaults_to_answered():
    assert _parse_abstention_verdict("the sky is blue") is False


# --- abstention_judge scorer (offline via mockllm) -----------------------


def _judge_state(question: str, answer: str, *, should_abstain: bool, category: str):
    return SimpleNamespace(
        input_text=question,
        output=SimpleNamespace(completion=answer),
        metadata={"should_abstain": should_abstain, "category": category},
    )


def _mock_judge(verdict: str):
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", verdict)],
    )


def test_judge_correct_when_abstains_on_abstain_row():
    state = _judge_state(
        "How tall is the tower?", "Which tower do you mean?",
        should_abstain=True, category="underspecified",
    )
    score = asyncio.run(
        abstention_judge(model=_mock_judge("VERDICT: ABSTAIN"))(state, None)
    )
    assert score.value == CORRECT
    assert score.metadata == {
        "should_abstain": True, "did_abstain": True, "category": "underspecified",
    }


def test_judge_incorrect_when_answers_an_abstain_row():
    state = _judge_state(
        "How tall is the tower?", "The tower is 300 metres tall.",
        should_abstain=True, category="underspecified",
    )
    score = asyncio.run(
        abstention_judge(model=_mock_judge("VERDICT: ANSWER"))(state, None)
    )
    assert score.value == INCORRECT
    assert score.metadata["did_abstain"] is False


def test_judge_correct_when_answers_a_control_row():
    state = _judge_state(
        "When was Barack Obama born?", "August 4, 1961.",
        should_abstain=False, category="control",
    )
    score = asyncio.run(
        abstention_judge(model=_mock_judge("VERDICT: ANSWER"))(state, None)
    )
    assert score.value == CORRECT
    assert score.metadata["did_abstain"] is False


# --- dataset schema & coverage -------------------------------------------

_DATASET = Path(__file__).parent.parent / "wiki_eval" / "datasets" / "abstention.jsonl"


def _load_rows():
    with _DATASET.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dataset_schema_and_counts():
    rows = _load_rows()
    assert len(rows) == 36
    for row in rows:
        assert set(row) == {"input", "target", "should_abstain", "category"}
        assert isinstance(row["input"], str) and row["input"].strip()
        assert isinstance(row["target"], str) and row["target"].strip()
        assert isinstance(row["should_abstain"], bool)
        assert isinstance(row["category"], str) and row["category"].strip()

    abstain = [r for r in rows if r["should_abstain"]]
    controls = [r for r in rows if not r["should_abstain"]]
    assert len(abstain) == 30
    assert len(controls) == 6
    assert all(r["category"] == "control" for r in controls)

    counts = Counter(r["category"] for r in abstain)
    assert counts == {
        "false_premise": 5,
        "unknowable": 5,
        "stale_realtime": 5,
        "underspecified": 4,
        "subjective": 3,
        "garbled": 4,
        "out_of_scope": 4,
    }
