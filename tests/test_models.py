"""Tests for data model classes."""

import pytest
from praisonai_editor.models import (
    ProbeResult, Word, TranscriptResult, Segment, EditPlan, EditResult,
    _format_srt_time,
)


def test_probe_result_basic():
    r = ProbeResult(path="/test.mp3", duration=120.5)
    assert r.path == "/test.mp3"
    assert r.duration == 120.5
    assert r.is_audio_only is True
    assert r.has_video is False


def test_probe_result_video():
    r = ProbeResult(path="/test.mp4", duration=60.0, has_video=True, video_codec="h264")
    assert r.is_audio_only is False
    d = r.to_dict()
    assert d["has_video"] is True
    assert d["video_codec"] == "h264"


def test_word():
    w = Word(text="hello", start=1.0, end=1.5, confidence=0.99)
    d = w.to_dict()
    assert d["text"] == "hello"
    assert d["start"] == 1.0
    assert d["end"] == 1.5


def test_transcript_result_text():
    t = TranscriptResult(text="Hello world", language="en", duration=5.0)
    assert t.text == "Hello world"
    d = t.to_dict()
    assert d["language"] == "en"


def test_default_openai_transcription_model_export():
    from praisonai_editor import DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    assert DEFAULT_OPENAI_TRANSCRIPTION_MODEL == "whisper-1"


def test_transcript_result_from_dict_roundtrip():
    words = [Word(text="Hi", start=0.1, end=0.3, confidence=0.9)]
    t = TranscriptResult(text="Hi", words=words, language="en", duration=1.0)
    t2 = TranscriptResult.from_dict(t.to_dict())
    assert t2.text == "Hi"
    assert len(t2.words) == 1
    assert t2.words[0].text == "Hi"
    assert t2.words[0].start == 0.1
    assert t2.duration == 1.0


def test_transcript_to_srt():
    words = [
        Word(text="Hello", start=0.0, end=0.5),
        Word(text="world.", start=0.6, end=1.0),
        Word(text="How", start=2.0, end=2.3),
        Word(text="are", start=2.3, end=2.5),
        Word(text="you?", start=2.5, end=3.0),
    ]
    t = TranscriptResult(text="Hello world. How are you?", words=words, duration=3.0)
    srt = t.to_srt()
    assert "1\n" in srt
    assert "Hello world." in srt
    assert "-->" in srt


def test_transcript_to_srt_empty():
    t = TranscriptResult(text="", words=[], duration=0.0)
    assert t.to_srt() == ""


def test_segment():
    s = Segment(
        start=0.0, end=1.0, action="remove",
        reason="filler", category="filler", text="um",
    )
    assert s.action == "remove"
    d = s.to_dict()
    assert d["category"] == "filler"


def test_edit_plan():
    segs = [
        Segment(start=0, end=5, action="keep", reason="content", category="content"),
        Segment(start=5, end=7, action="remove", reason="silence", category="silence"),
        Segment(start=7, end=10, action="keep", reason="content", category="content"),
    ]
    plan = EditPlan(
        segments=segs, original_duration=10.0,
        edited_duration=8.0, removed_duration=2.0,
    )
    assert len(plan.get_keep_segments()) == 2
    assert len(plan.get_remove_segments()) == 1
    d = plan.to_dict()
    assert d["original_duration"] == 10.0


def test_edit_result():
    r = EditResult(input_path="/in.mp3", output_path="/out.mp3", success=True)
    d = r.to_dict()
    assert d["success"] is True
    assert d["input_path"] == "/in.mp3"


def test_format_srt_time():
    assert _format_srt_time(0.0) == "00:00:00,000"
    assert _format_srt_time(3661.5) == "01:01:01,500"
    assert _format_srt_time(90.123) == "00:01:30,123"
