"""Unit test for the multilingual record->Sample mapping (no API/network)."""

from wiki_eval.tasks import _multilingual_record_to_sample


def test_record_maps_metadata_fields_into_sample():
    record = {
        "input": "Where was Otar Tevdoradze born?",
        "target": "Kutaisi.",
        "category": "richer_native_page",
        "language": "ka",
        "language_name": "Georgian",
        "hop_type": "single",
    }
    sample = _multilingual_record_to_sample(record)
    assert sample.input == "Where was Otar Tevdoradze born?"
    assert sample.target == "Kutaisi."
    assert sample.metadata == {
        "category": "richer_native_page",
        "language": "ka",
        "language_name": "Georgian",
        "hop_type": "single",
    }
