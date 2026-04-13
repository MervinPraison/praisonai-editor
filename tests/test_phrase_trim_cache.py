"""Sidecar transcript cache for phrase trim."""

import json

from praisonai_editor.phrase_trim import (
    _try_load_transcript_cache,
    _write_transcript_cache,
    transcript_sidecar_path,
)
from praisonai_editor.models import TranscriptResult, Word


def test_transcript_sidecar_naming(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    assert transcript_sidecar_path(media).name == "talk.mp3.praisonai.transcript.json"


def test_cache_roundtrip(tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"abc")
    tr = TranscriptResult(
        text="hello",
        words=[Word(text="hello", start=0.0, end=0.5)],
        language="en",
        duration=1.0,
    )
    _write_transcript_cache(media, tr)
    loaded = _try_load_transcript_cache(media)
    assert loaded is not None
    assert loaded.text == "hello"
    assert len(loaded.words) == 1


def test_cache_invalidated_on_size_change(tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"abc")
    tr = TranscriptResult(text="x", words=[], duration=0.0)
    _write_transcript_cache(media, tr)
    media.write_bytes(b"abcd")
    assert _try_load_transcript_cache(media) is None


def test_cache_rejects_wrong_version(tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"x")
    side = transcript_sidecar_path(media)
    side.write_text(
        json.dumps(
            {
                "_praisonai_cache_version": 999,
                "_praisonai_audio_path": str(media.resolve()),
                "_praisonai_audio_mtime_ns": media.stat().st_mtime_ns,
                "_praisonai_audio_size": media.stat().st_size,
                "text": "",
                "words": [],
                "language": "en",
                "duration": 0.0,
            }
        ),
        encoding="utf-8",
    )
    assert _try_load_transcript_cache(media) is None
