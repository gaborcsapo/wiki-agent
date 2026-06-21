"""Tests for the multilingual correctness scorer (pure; no API/network).

The scorer wraps Inspect's model_graded_qa, which needs a model at *score*
time only; constructing the scorer and inspecting its template is offline-safe.
"""

from wiki_eval.scorers import _MULTILINGUAL_QA_TEMPLATE, multilingual_correctness


def test_template_keeps_required_grading_placeholders():
    # model_graded_qa fills these four fields; dropping any breaks grading.
    for field in ("{question}", "{answer}", "{criterion}", "{instructions}"):
        assert field in _MULTILINGUAL_QA_TEMPLATE


def test_template_instructs_language_neutral_grading():
    text = _MULTILINGUAL_QA_TEMPLATE.lower()
    assert "any language" in text


def test_scorer_constructs_offline():
    # No ANTHROPIC_API_KEY needed: model is resolved lazily at score time.
    assert callable(multilingual_correctness())
