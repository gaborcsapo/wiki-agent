"""Unit test for the FRAMES record->Sample mapping (no API/network)."""

from wiki_eval.tasks import _frames_record_to_sample


def test_frames_record_maps_reference_pages_into_metadata():
    record = {
        "input": "Who walked on the Moon first?",
        "target": "Neil Armstrong",
        "reference_pages": ["https://en.wikipedia.org/wiki/Apollo_11"],
    }
    sample = _frames_record_to_sample(record)
    assert sample.input == "Who walked on the Moon first?"
    assert sample.target == "Neil Armstrong"
    assert sample.metadata["reference_pages"] == ["https://en.wikipedia.org/wiki/Apollo_11"]


def test_frames_record_defaults_missing_reference_pages_to_empty():
    sample = _frames_record_to_sample({"input": "Q", "target": "A"})
    assert sample.metadata["reference_pages"] == []
