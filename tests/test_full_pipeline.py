"""Tests for full pipeline phase definitions."""

from praisonai_editor.full_pipeline import FULL_PIPELINE, describe_full_pipeline, validate_stem


def test_full_pipeline_has_biblerevelation_phases():
    ids = [p.id for p in FULL_PIPELINE]
    assert "9" in ids
    assert any("biblerevelation" in p.name.lower() for p in FULL_PIPELINE)


def test_describe_full_pipeline_mentions_publish():
    text = describe_full_pipeline()
    assert "full pipeline" in text.lower()
    assert "biblerevelation" in text.lower()


def test_validate_stem_missing(tmp_path):
    result = validate_stem("nonexistent_stem", repo=tmp_path)
    assert result.complete is False
    assert any(not p["ok"] for p in result.phases)
