"""Dataset-integrity test for multilingual_qa.jsonl (pure; no API/network)."""

import json
from collections import Counter
from pathlib import Path

import pytest

_DATASET = Path(__file__).parents[1] / "wiki_eval" / "datasets" / "multilingual_qa.jsonl"

_CATEGORIES = {"cross_lingual_fact", "richer_native_page", "foreign_language_query"}
_HOP_TYPES = {"single", "multi"}
# language code -> display name, the languages mined for this benchmark.
_LANGUAGES = {
    "hu": "Hungarian", "is": "Icelandic", "et": "Estonian", "sw": "Swahili",
    "cy": "Welsh", "eu": "Basque", "ka": "Georgian", "hy": "Armenian", "yo": "Yoruba",
}


def _rows():
    with _DATASET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_dataset_has_thirty_rows():
    assert len(_rows()) == 30


@pytest.mark.parametrize("row", _rows())
def test_row_is_well_formed(row):
    assert row["input"].strip(), "input must be non-empty"
    assert row["target"].strip(), "target must be non-empty"
    assert row["category"] in _CATEGORIES
    assert row["hop_type"] in _HOP_TYPES
    assert row["language"] in _LANGUAGES
    assert row["language_name"] == _LANGUAGES[row["language"]]


def test_every_category_present():
    seen = {row["category"] for row in _rows()}
    assert seen == _CATEGORIES


def test_hungarian_is_most_represented():
    counts = Counter(row["language"] for row in _rows())
    top_lang, _ = counts.most_common(1)[0]
    assert top_lang == "hu"
