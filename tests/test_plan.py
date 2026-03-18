"""Tests for edit plan creation."""

import pytest
from praisonai_editor.models import Word, TranscriptResult
from praisonai_editor.plan import (
    detect_fillers, detect_repetitions, detect_silences,
    HeuristicEditor, PRESETS, create_edit_plan,
)


def make_words(data):
    """Helper to create Word list from [(text, start, end), ...]."""
    return [Word(text=t, start=s, end=e) for t, s, e in data]


class TestDetectFillers:
    def test_finds_fillers(self):
        words = make_words([
            ("um", 0.0, 0.3),
            ("hello", 0.4, 0.8),
            ("uh", 1.0, 1.2),
            ("world", 1.3, 1.7),
        ])
        segments = detect_fillers(words)
        assert len(segments) == 2
        assert all(s.category == "filler" for s in segments)

    def test_no_fillers(self):
        words = make_words([("hello", 0.0, 0.5), ("world", 0.6, 1.0)])
        segments = detect_fillers(words)
        assert len(segments) == 0


class TestDetectRepetitions:
    def test_finds_repetitions(self):
        words = make_words([
            ("the", 0.0, 0.2),
            ("the", 0.3, 0.5),
            ("cat", 0.6, 0.9),
        ])
        segments = detect_repetitions(words)
        assert len(segments) == 1
        assert segments[0].category == "repetition"

    def test_no_repetitions(self):
        words = make_words([("hello", 0.0, 0.5), ("world", 0.6, 1.0)])
        segments = detect_repetitions(words)
        assert len(segments) == 0


class TestDetectSilences:
    def test_finds_gap_silence(self):
        words = make_words([
            ("hello", 0.0, 0.5),
            ("world", 3.0, 3.5),
        ])
        segments = detect_silences(words, 5.0, min_silence=1.5)
        assert len(segments) >= 1
        assert any(s.category == "silence" for s in segments)

    def test_no_silence(self):
        words = make_words([
            ("hello", 0.0, 0.5),
            ("world", 0.6, 1.0),
        ])
        segments = detect_silences(words, 1.0, min_silence=1.5)
        assert len(segments) == 0


class TestHeuristicEditor:
    def test_creates_plan(self):
        words = make_words([
            ("um", 0.0, 0.3),
            ("hello", 0.4, 0.8),
            ("world", 4.0, 4.5),  # 3.2s gap after "hello"
        ])
        transcript = TranscriptResult(
            text="um hello world", words=words, duration=5.0,
        )
        editor = HeuristicEditor()
        plan = editor.create_plan(transcript, 5.0)
        assert plan.original_duration == 5.0
        assert plan.edited_duration < plan.original_duration
        assert plan.removed_duration > 0

    def test_all_disabled(self):
        words = make_words([
            ("um", 0.0, 0.3),
            ("hello", 0.4, 0.8),
        ])
        transcript = TranscriptResult(text="um hello", words=words, duration=1.0)
        editor = HeuristicEditor()
        plan = editor.create_plan(
            transcript, 1.0,
            remove_fillers=False, remove_repetitions=False, remove_silence=False,
        )
        assert plan.removed_duration == 0


class TestPresets:
    def test_podcast_preset(self):
        assert "podcast" in PRESETS
        assert PRESETS["podcast"]["remove_fillers"] is True

    def test_meeting_preset(self):
        assert "meeting" in PRESETS
        assert PRESETS["meeting"]["remove_repetitions"] is False


class TestCreateEditPlan:
    def test_with_preset(self):
        words = make_words([("hello", 0.0, 0.5)])
        transcript = TranscriptResult(text="hello", words=words, duration=1.0)
        plan = create_edit_plan(transcript, 1.0, preset="podcast")
        assert plan.original_duration == 1.0
