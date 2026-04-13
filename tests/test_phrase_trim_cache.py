"""Transcript cache for phrase trim."""

import json

import pytest

from praisonai_editor.phrase_trim import (
    _media_cache_dir_name,
    _try_load_transcript_cache,
    _upgrade_short_digest_cache_dir,
    _write_transcript_cache,
    transcript_cache_file,
    transcript_sidecar_path,
)
from praisonai_editor.models import TranscriptResult, Word


@pytest.fixture
def editor_cache_home(tmp_path, monkeypatch):
    root = tmp_path / "editor"
    monkeypatch.setattr("praisonai_editor.phrase_trim._editor_cache_root", lambda: root)
    return root


def test_transcript_cache_file_under_home(editor_cache_home, tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    p = transcript_cache_file(media)
    assert p.name == "transcript.json"
    assert p.parent.parent == editor_cache_home
    stem, hexd = p.parent.name.rsplit("_", 1)
    assert stem == "talk"
    assert len(hexd) == 64
    assert all(c in "0123456789abcdef" for c in hexd)


def test_legacy_sidecar_naming(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    assert transcript_sidecar_path(media).name == "talk.mp3.praisonai.transcript.json"


def test_cache_roundtrip(editor_cache_home, tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"abc")
    tr = TranscriptResult(
        text="hello",
        words=[Word(text="hello", start=0.0, end=0.5)],
        language="en",
        duration=1.0,
    )
    _write_transcript_cache(media, tr)
    loaded, path = _try_load_transcript_cache(media)
    assert loaded is not None
    assert path == transcript_cache_file(media)
    assert loaded.text == "hello"
    assert len(loaded.words) == 1


def test_cache_invalidated_on_size_change(editor_cache_home, tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"abc")
    tr = TranscriptResult(text="x", words=[], duration=0.0)
    _write_transcript_cache(media, tr)
    media.write_bytes(b"abcd")
    assert _try_load_transcript_cache(media) == (None, None)


def test_cache_rejects_wrong_version(editor_cache_home, tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"x")
    cache = transcript_cache_file(media)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
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
    assert _try_load_transcript_cache(media) == (None, None)


def test_short_digest_folder_renamed_on_upgrade(editor_cache_home, tmp_path):
    media = tmp_path / "clip.mp3"
    media.write_bytes(b"body")
    st = media.stat()
    short_dir = editor_cache_home / _media_cache_dir_name(media, digest_chars=12)
    short_dir.mkdir(parents=True)
    short_file = short_dir / "transcript.json"
    short_file.write_text(
        json.dumps(
            {
                "_praisonai_cache_version": 1,
                "_praisonai_audio_path": str(media.resolve()),
                "_praisonai_audio_mtime_ns": st.st_mtime_ns,
                "_praisonai_audio_size": st.st_size,
                "text": "ok",
                "words": [{"text": "ok", "start": 0.0, "end": 0.2, "confidence": 1.0}],
                "language": "en",
                "duration": 0.2,
            }
        ),
        encoding="utf-8",
    )
    _upgrade_short_digest_cache_dir(media)
    long_dir = transcript_cache_file(media).parent
    assert long_dir.is_dir()
    assert (long_dir / "transcript.json").is_file()
    assert not short_dir.exists()
    loaded, path = _try_load_transcript_cache(media)
    assert loaded is not None
    assert path == transcript_cache_file(media)


def test_legacy_sidecar_still_read(editor_cache_home, tmp_path):
    """Primary missing: load legacy file next to media."""
    media = tmp_path / "legacy.mp3"
    media.write_bytes(b"xyz")
    side = transcript_sidecar_path(media)
    st = media.stat()
    side.write_text(
        json.dumps(
            {
                "_praisonai_cache_version": 1,
                "_praisonai_audio_path": str(media.resolve()),
                "_praisonai_audio_mtime_ns": st.st_mtime_ns,
                "_praisonai_audio_size": st.st_size,
                "text": "hi",
                "words": [{"text": "hi", "start": 0.0, "end": 0.1, "confidence": 1.0}],
                "language": "en",
                "duration": 0.1,
            }
        ),
        encoding="utf-8",
    )
    loaded, path = _try_load_transcript_cache(media)
    assert loaded is not None
    assert path == side
