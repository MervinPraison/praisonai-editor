"""Transcript cache for phrase trim."""

import json
import shutil
import subprocess

import pytest

from praisonai_editor.phrase_trim import (
    _exclusive_end_phrase_first_word_time,
    _first_phrase_first_word_time,
    _media_cache_dir_name,
    _norm,
    _phrase_match_starts,
    _try_load_transcript_cache,
    _upgrade_short_digest_cache_dir,
    _write_transcript_cache,
    trim_between_phrase_markers,
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


def test_phrase_first_end_exclusive_time():
    words = [
        Word(text="already", start=1.0, end=1.2),
        Word(text="you", start=1.2, end=1.4),
        Word(text="Our", start=1.5, end=2.0),
        Word(text="Heavenly", start=2.0, end=2.5),
        Word(text="Father", start=2.5, end=3.0),
    ]
    p = _norm("our heavenly father")
    t = _exclusive_end_phrase_first_word_time(words, p, max_span=10, end_last_match=True)
    assert t == 1.5


def test_phrase_first_start_includes_first_spoken_word_of_phrase():
    words = [
        Word(text="skip", start=0.0, end=0.4),
        Word(text="so", start=0.5, end=0.7),
        Word(text="what", start=0.7, end=1.0),
        Word(text="topic", start=1.0, end=1.2),
    ]
    p = _norm("so what topic")
    t = _first_phrase_first_word_time(words, p, max_span=10)
    assert t == 0.5


class TestOutOfOrderWordRobustness:
    """Real transcripts occasionally have an out-of-order word -- a word
    whose reported `start` is earlier than the PREVIOUS word's own `start`
    (an ASR timestamp glitch confirmed present in real .transcript.json
    output, distinct from a zero-duration word). `_exclusive_end_phrase_
    first_word_time` and `trim_between_phrase_markers`'s own "window"
    branch both need to pick the LAST occurrence of the end phrase in
    TRANSCRIPT (list) order, not whichever candidate happens to have the
    numerically largest `start` value -- those are only the same thing
    when every word's timestamp is monotonically increasing.
    """

    def test_exclusive_end_time_uses_transcript_order_not_max_value(self):
        # "stop now" occurs twice. The second (true last, later in the
        # list) occurrence has a corrupted start (9.9) that is LOWER than
        # the first occurrence's own start (10.0) -- max() would silently
        # return the FIRST occurrence's time instead of the last one's.
        words = [
            Word(text="stop", start=10.0, end=10.3),
            Word(text="now", start=10.3, end=10.6),
            Word(text="ok", start=10.6, end=10.8),
            Word(text="stop", start=9.9, end=10.9),  # out-of-order
            Word(text="now", start=10.9, end=11.2),
        ]
        p = _norm("stop now")
        t = _exclusive_end_phrase_first_word_time(words, p, max_span=10, end_last_match=True)
        # The true last occurrence starts at 9.9s -- NOT 10.0s (what a
        # max()-over-values selection would have wrongly returned).
        assert t == 9.9

    def test_exclusive_end_time_first_match_still_works_when_ordered(self):
        # Sanity: normal, monotonically-ordered data is unaffected.
        words = [
            Word(text="stop", start=10.0, end=10.3),
            Word(text="now", start=10.3, end=10.6),
            Word(text="ok", start=10.6, end=10.8),
            Word(text="stop", start=12.0, end=12.3),
            Word(text="now", start=12.3, end=12.6),
        ]
        p = _norm("stop now")
        assert _exclusive_end_phrase_first_word_time(
            words, p, max_span=10, end_last_match=True
        ) == 12.0
        assert _exclusive_end_phrase_first_word_time(
            words, p, max_span=10, end_last_match=False
        ) == 10.0


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestTrimBetweenPhraseMarkersOutOfOrderWord:
    """Real ffmpeg, a real generated audio file, and a hand-built
    transcript (via `transcript_path=`, so no ASR call is made) with a
    genuine out-of-order word -- proves `trim_between_phrase_markers`'s
    default ("window") boundary mode cuts at the TRUE last occurrence of
    the end phrase, not at whatever candidate has the largest raw
    timestamp value."""

    def test_window_mode_cuts_at_the_true_last_occurrence(self, tmp_path):
        ffmpeg = shutil.which("ffmpeg")
        media = tmp_path / "in.wav"
        result = subprocess.run(
            [ffmpeg, "-y", "-nostdin", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "12.0", str(media)],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()[-800:]

        # "banana rocket" occurs twice. The second (true last) occurrence's
        # first word ("banana") is out-of-order -- its start (4.9) is
        # earlier than the immediately preceding word's own start (7.0).
        words = [
            Word(text="hello", start=0.0, end=0.5),
            Word(text="apple", start=1.0, end=1.3),
            Word(text="banana", start=5.0, end=5.3),
            Word(text="rocket", start=5.3, end=5.6),
            Word(text="carrot", start=7.0, end=7.3),
            Word(text="banana", start=4.9, end=9.0),  # out-of-order
            Word(text="rocket", start=9.0, end=9.6),
        ]
        transcript = TranscriptResult(
            text=" ".join(w.text for w in words), words=words, language="en", duration=12.0
        )
        # Confirm the fixture actually reproduces the hazard this test
        # guards against: transcript-order-vs-value disagree.
        starts = _phrase_match_starts(words, _norm("banana rocket"))
        assert max(starts) != starts[-1]

        tpath = tmp_path / "transcript.json"
        tpath.write_text(json.dumps(transcript.to_dict()), encoding="utf-8")

        out = tmp_path / "out.wav"
        trim_between_phrase_markers(
            str(media), str(out),
            start_phrase="hello",
            end_phrase="banana rocket",
            end_last_match=True,
            transcript_path=str(tpath),
            refine_with_openai=False,
        )

        assert out.exists()
        probe = subprocess.run(
            [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip())
        # Cut should run from 0.0s to the TRUE last occurrence's start
        # (4.9s), not to 7.0s (the wrong value a plain max() selection
        # over candidate timestamps would have picked).
        assert duration == pytest.approx(4.9, abs=0.2)
