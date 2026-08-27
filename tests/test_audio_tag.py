"""Tests for the source_audio tagging/resolving feature (audio_tag.py).

A transcript can end up separated from the audio it was made from -- moved
to a different folder, handed to a tool that only accepts one file, etc.
These tests exercise the real matching logic: content-fingerprint (exact),
filename-only fallback, and duration-only fallback -- using real files on
disk, not mocked I/O, since the whole point is hashing/stat'ing real bytes.
"""

from __future__ import annotations

import json

import pytest

from praisonai_editor.audio_tag import (
    compute_audio_tag,
    find_matching_audio,
    tag_source_audio,
)
from praisonai_editor.models import TranscriptResult, Word


def _write_audio(path, content=b"fake-wav-bytes-not-real-audio"):
    path.write_bytes(content)
    return str(path)


def test_compute_audio_tag_has_stable_fields(tmp_path):
    audio = _write_audio(tmp_path / "song.wav", b"abc" * 1000)
    tag = compute_audio_tag(audio)
    assert tag["filename"] == "song.wav"
    assert tag["size_bytes"] == 3000
    assert isinstance(tag["sha256_8mb"], str) and len(tag["sha256_8mb"]) == 16
    # Same bytes -> same fingerprint, deterministic.
    assert compute_audio_tag(audio) == tag


def test_different_content_gives_different_fingerprint(tmp_path):
    a = _write_audio(tmp_path / "a.wav", b"content-a" * 500)
    b = _write_audio(tmp_path / "b.wav", b"content-b" * 500)
    assert compute_audio_tag(a)["sha256_8mb"] != compute_audio_tag(b)["sha256_8mb"]


def test_tag_source_audio_stamps_result_in_place(tmp_path):
    audio = _write_audio(tmp_path / "clip.mp3")
    result = TranscriptResult(text="hi", words=[Word("hi", 0.0, 0.5)], duration=0.5)
    assert result.source_audio is None
    returned = tag_source_audio(result, audio)
    assert returned is result
    assert result.source_audio["filename"] == "clip.mp3"


def test_tag_source_audio_silently_skips_missing_file(tmp_path):
    result = TranscriptResult(text="hi")
    tag_source_audio(result, str(tmp_path / "does-not-exist.wav"))
    assert result.source_audio is None


def test_transcript_round_trip_preserves_source_audio(tmp_path):
    audio = _write_audio(tmp_path / "voice.wav")
    tag = compute_audio_tag(audio)
    tr = TranscriptResult(text="hi", words=[Word("hi", 0.0, 1.0)], duration=1.0, source_audio=tag)
    restored = TranscriptResult.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert restored.source_audio == tag


def test_old_transcript_json_without_tag_is_unaffected():
    old = {"text": "legacy", "words": [], "language": "en", "duration": 1.0}
    tr = TranscriptResult.from_dict(old)
    assert tr.source_audio is None
    assert "source_audio" not in tr.to_dict()


def test_find_matching_audio_exact_match(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    audio = _write_audio(real_dir / "episode.wav", b"real-content" * 2000)
    tag = compute_audio_tag(audio)

    candidates = find_matching_audio(tag, [str(real_dir)])
    assert len(candidates) == 1
    assert candidates[0]["confidence"] == "exact"
    assert candidates[0]["path"] == str((real_dir / "episode.wav").resolve())


def test_find_matching_audio_searches_multiple_dirs_and_skips_missing(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    audio = _write_audio(real_dir / "episode.wav", b"real-content" * 2000)
    tag = compute_audio_tag(audio)

    candidates = find_matching_audio(
        tag, [str(empty_dir), str(tmp_path / "nonexistent"), str(real_dir)]
    )
    assert len(candidates) == 1
    assert candidates[0]["confidence"] == "exact"


def test_find_matching_audio_filename_fallback_when_content_differs(tmp_path):
    original_dir = tmp_path / "original"
    original_dir.mkdir()
    original = _write_audio(original_dir / "episode.wav", b"version-one" * 1000)
    tag = compute_audio_tag(original)

    # Same filename, elsewhere, re-encoded (different bytes) -- fingerprint
    # won't match, but the filename is still a real (weaker) signal.
    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    _write_audio(moved_dir / "episode.wav", b"version-two-reencoded" * 900)

    candidates = find_matching_audio(tag, [str(moved_dir)])
    assert len(candidates) == 1
    assert candidates[0]["confidence"] == "filename"


def test_find_matching_audio_no_match_returns_empty(tmp_path):
    d = tmp_path / "unrelated"
    d.mkdir()
    _write_audio(d / "something-else.mp3", b"whatever")
    tag = {"filename": "not-here.wav", "size_bytes": 123, "sha256_8mb": "0" * 16}

    assert find_matching_audio(tag, [str(d)]) == []


def test_find_matching_audio_ignores_non_audio_extensions(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    tag = {"filename": "clip.txt", "size_bytes": 5, "sha256_8mb": "0" * 16}
    (d / "clip.txt").write_text("hello")
    # .txt isn't in AUDIO_EXTS, so even an identical filename must not match.
    assert find_matching_audio(tag, [str(d)]) == []


def test_find_matching_audio_empty_tag_returns_empty(tmp_path):
    assert find_matching_audio({}, [str(tmp_path)]) == []
