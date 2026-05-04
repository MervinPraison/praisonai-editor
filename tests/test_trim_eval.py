"""Tests for trim eval (mocked ASR and ffmpeg extracts)."""

from pathlib import Path

import pytest

from praisonai_editor.models import TranscriptResult
from praisonai_editor.trim_eval import (
    _try_load_eval_cache,
    _write_eval_cache,
    evaluate_trim_edges,
)


@pytest.fixture
def tiny_mp3(tmp_path):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"fake")
    return p


@pytest.fixture
def eval_cache_root(tmp_path, monkeypatch):
    root = tmp_path / "editor"
    monkeypatch.setattr("praisonai_editor.trim_eval._editor_cache_root", lambda: root)
    return root


def test_eval_cache_roundtrip(tmp_path, eval_cache_root):
    parent = tmp_path / "sermon.mp3"
    parent.write_bytes(b"body")
    tr = TranscriptResult(text="cached line", words=[], language="en", duration=1.0)
    _write_eval_cache(parent, 0.0, 20.0, "head", "en", "whisper-1", tr)
    loaded = _try_load_eval_cache(parent, 0.0, 20.0, "head", "en", "whisper-1")
    assert loaded is not None
    assert loaded.text == "cached line"


def test_evaluate_trim_edges_merges_context_for_checks(monkeypatch, tiny_mp3, eval_cache_root):
    def fake_probe(_path):
        class P:
            duration = 100.0

        return P()

    def fake_transcribe(path, **kwargs):
        name = Path(path).name
        texts = {
            "head_segment.mp3": "intro so what topic are we seeing",
            "tail_segment.mp3": "he has redeemed you",
            "head_before_edge.mp3": "short edge",
            "head_after.mp3": "continuation after head",
            "tail_before.mp3": "lead in before tail block",
            "tail_after_suffix.mp3": "last words",
        }
        return TranscriptResult(
            text=texts.get(name, ""),
            words=[],
            language="en",
            duration=1.0,
        )

    monkeypatch.setattr("praisonai_editor.trim_eval.probe_media", fake_probe)
    monkeypatch.setattr("praisonai_editor.trim_eval._ffmpeg_extract_segment", lambda *a, **k: None)
    monkeypatch.setattr("praisonai_editor.trim_eval.transcribe_audio", fake_transcribe)

    r = evaluate_trim_edges(
        tiny_mp3,
        head_context_before_sec=5.0,
        head_context_after_sec=10.0,
        tail_context_before_sec=10.0,
        tail_context_after_sec=3.0,
        head_contains="so what topic are we seeing",
        tail_forbid=["our heavenly father"],
    )
    assert r.ok
    assert "so what topic" in r.head_transcript.lower()
    assert r.head_before_transcript
    assert r.head_after_transcript
    assert r.tail_before_transcript
    assert r.tail_after_transcript
    assert not r.failures


def test_ai_judge_merges_into_ok(monkeypatch, tiny_mp3, eval_cache_root):
    def fake_probe(_path):
        class P:
            duration = 40.0

        return P()

    def fake_transcribe(path, **kwargs):
        return TranscriptResult(text="ok", words=[], language="en", duration=1.0)

    def fake_judge(**kwargs):
        return True, "looks fine"

    monkeypatch.setattr("praisonai_editor.trim_eval.probe_media", fake_probe)
    monkeypatch.setattr("praisonai_editor.trim_eval._ffmpeg_extract_segment", lambda *a, **k: None)
    monkeypatch.setattr("praisonai_editor.trim_eval.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("praisonai_editor.trim_eval._ai_judge_trim_regions", fake_judge)

    r = evaluate_trim_edges(
        tiny_mp3,
        head_context_before_sec=0.0,
        head_context_after_sec=0.0,
        tail_context_before_sec=0.0,
        tail_context_after_sec=0.0,
        ai_judge=True,
        ai_start_intent="open with topic",
        ai_end_intent="end before prayer",
    )
    assert r.ok
    assert r.ai_judge_ran
    assert r.ai_judge_acceptable is True


def test_tail_forbid_fails_when_present(monkeypatch, tiny_mp3, eval_cache_root):
    def fake_probe(_path):
        class P:
            duration = 50.0

        return P()

    def fake_transcribe(path, **kwargs):
        name = Path(path).name
        if name == "tail_segment.mp3":
            return TranscriptResult(
                text="our heavenly father we pray",
                words=[],
                language="en",
                duration=1.0,
            )
        return TranscriptResult(text="x", words=[], language="en", duration=1.0)

    monkeypatch.setattr("praisonai_editor.trim_eval.probe_media", fake_probe)
    monkeypatch.setattr("praisonai_editor.trim_eval._ffmpeg_extract_segment", lambda *a, **k: None)
    monkeypatch.setattr("praisonai_editor.trim_eval.transcribe_audio", fake_transcribe)

    r = evaluate_trim_edges(
        tiny_mp3,
        head_context_before_sec=0.0,
        head_context_after_sec=0.0,
        tail_context_before_sec=0.0,
        tail_context_after_sec=0.0,
        tail_forbid=["our heavenly father"],
    )
    assert not r.ok
    assert r.failures
