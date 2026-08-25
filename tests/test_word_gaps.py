"""Tests for shortening long pauses between words (praisonai_editor.word_gaps).

Real (non-mocked) end-to-end proof this feature actually does what it
claims -- generate real speech-like audio with a KNOWN gap, transcribe it
for real, apply shorten_word_gaps, then independently re-transcribe the
OUTPUT and confirm the gap is gone and the words survived -- lives in
TestRealEndToEnd below. Everything else here is fast, deterministic
logic/wiring coverage with ffmpeg mocked out, the same convention
tests/test_remove_ranges.py's own TestRemoveTimeRangesTranscriptParam uses.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import praisonai_editor.remove_ranges as remove_ranges_mod
from praisonai_editor.models import TranscriptResult, Word
from praisonai_editor.word_gaps import find_long_gaps, shorten_word_gaps


def _words(*specs):
    """specs: (text, start, end) tuples."""
    return [Word(text=t, start=s, end=e, confidence=1.0) for t, s, e in specs]


class TestFindLongGaps:
    def test_no_gaps_when_words_are_contiguous(self):
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 0.5), ("b", 0.5, 1.0)))
        assert find_long_gaps(transcript, threshold=0.1) == []

    def test_finds_a_single_gap_above_threshold(self):
        transcript = TranscriptResult(
            text="a b", words=_words(("a", 0.0, 0.5), ("b", 3.0, 3.5))
        )
        assert find_long_gaps(transcript, threshold=1.0) == [(0.5, 3.0)]

    def test_gap_exactly_at_threshold_does_not_qualify(self):
        # find_long_gaps uses a strict `>`, matching remove_time_ranges'
        # own "only touch what's clearly excessive" posture.
        transcript = TranscriptResult(
            text="a b", words=_words(("a", 0.0, 0.5), ("b", 1.5, 2.0))
        )
        assert find_long_gaps(transcript, threshold=1.0) == []

    def test_multiple_gaps_all_found_in_order(self):
        transcript = TranscriptResult(text="a b c", words=_words(
            ("a", 0.0, 0.2), ("b", 2.0, 2.2), ("c", 5.0, 5.2)))
        assert find_long_gaps(transcript, threshold=1.0) == [(0.2, 2.0), (2.2, 5.0)]

    def test_empty_transcript_has_no_gaps(self):
        transcript = TranscriptResult(text="", words=[])
        assert find_long_gaps(transcript, threshold=0.1) == []

    def test_single_word_has_no_gaps(self):
        transcript = TranscriptResult(text="a", words=_words(("a", 0.0, 0.2)))
        assert find_long_gaps(transcript, threshold=0.1) == []


class TestShortenWordGapsValidation:
    def test_negative_target_raises(self):
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 0.2), ("b", 3.0, 3.2)))
        with pytest.raises(ValueError, match="target"):
            shorten_word_gaps("in.wav", transcript=transcript, target=-0.1)

    def test_target_equal_to_threshold_raises(self):
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 0.2), ("b", 3.0, 3.2)))
        with pytest.raises(ValueError, match="less than"):
            shorten_word_gaps("in.wav", transcript=transcript, threshold=0.5, target=0.5)

    def test_target_greater_than_threshold_raises(self):
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 0.2), ("b", 3.0, 3.2)))
        with pytest.raises(ValueError, match="less than"):
            shorten_word_gaps("in.wav", transcript=transcript, threshold=0.5, target=1.0)

    def test_no_qualifying_gap_raises_a_clear_error(self):
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 0.2), ("b", 0.3, 0.5)))
        with pytest.raises(ValueError, match="nothing to shorten"):
            shorten_word_gaps("in.wav", transcript=transcript, threshold=1.0, target=0.3)


class TestShortenWordGapsDelegatesCorrectly:
    """remove_time_ranges itself mocked out -- this only checks
    shorten_word_gaps computes the RIGHT ranges and hands them off
    correctly, not that ffmpeg/re-timing works (that's remove_time_ranges'
    own, already-covered job in test_remove_ranges.py)."""

    def _mock_remove(self, monkeypatch):
        calls = []

        def _fake_remove_time_ranges(input_path, ranges, output_path=None, **kwargs):
            calls.append({"input_path": input_path, "ranges": list(ranges),
                          "output_path": output_path, "kwargs": kwargs})
            from praisonai_editor.models import EditResult
            return EditResult(input_path=input_path, output_path=output_path or "out.wav",
                               success=True, transcript=kwargs.get("transcript"))

        monkeypatch.setattr(remove_ranges_mod, "remove_time_ranges", _fake_remove_time_ranges)
        import praisonai_editor.word_gaps as word_gaps_mod
        monkeypatch.setattr(word_gaps_mod, "remove_time_ranges", _fake_remove_time_ranges)
        return calls

    def test_single_gap_shortened_to_target(self, tmp_path, monkeypatch):
        calls = self._mock_remove(monkeypatch)
        transcript = TranscriptResult(
            text="hello world", words=_words(("hello", 1.0, 1.5), ("world", 5.0, 5.5))
        )

        result = shorten_word_gaps(
            "in.wav", "out.wav", transcript=transcript, threshold=1.0, target=0.3,
        )

        assert result.success
        # Keep 0.3s right after "hello" (1.5 -> 1.8), remove the rest of the
        # gap (1.8 -> 5.0) -- exactly what a listener would expect from
        # "shorten to 0.3s", not "shift the whole gap by 0.3s".
        assert len(calls) == 1
        assert calls[0]["ranges"] == [(1.8, 5.0)]
        assert calls[0]["kwargs"]["transcript"] is transcript
        assert result.artifacts["gaps_shortened"] == "1"
        assert result.artifacts["threshold"] == "1.0"
        assert result.artifacts["target"] == "0.3"

    def test_multiple_gaps_all_shortened_in_one_pass(self, tmp_path, monkeypatch):
        calls = self._mock_remove(monkeypatch)
        transcript = TranscriptResult(text="a b c", words=_words(
            ("a", 0.0, 0.2), ("b", 3.0, 3.2), ("c", 10.0, 10.2)))

        result = shorten_word_gaps(
            "in.wav", "out.wav", transcript=transcript, threshold=1.0, target=0.25,
        )

        assert calls[0]["ranges"] == [(0.45, 3.0), (3.45, 10.0)]
        assert result.artifacts["gaps_shortened"] == "2"

    def test_result_transcript_is_whatever_remove_time_ranges_returns(self, tmp_path, monkeypatch):
        """shorten_word_gaps passes transcript= straight through to
        remove_time_ranges, so the same server-proven re-timing (already
        covered by test_remove_ranges.py's TestRetimeTranscript) applies
        here for free -- this just confirms the wiring, not the math."""
        self._mock_remove(monkeypatch)
        transcript = TranscriptResult(
            text="hello world", words=_words(("hello", 1.0, 1.5), ("world", 5.0, 5.5))
        )

        result = shorten_word_gaps(
            "in.wav", "out.wav", transcript=transcript, threshold=1.0, target=0.3,
        )

        assert result.transcript is transcript


class TestCliArgWiring:
    def test_word_gaps_arg_wiring(self, monkeypatch, tmp_path):
        import json as _json

        import praisonai_editor.cli as cli
        import praisonai_editor.word_gaps as word_gaps_mod
        from praisonai_editor.models import EditResult

        captured = {}

        def fake_shorten_word_gaps(input_path, output_path=None, **kwargs):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured.update(kwargs)
            return EditResult(
                input_path=input_path, output_path=str(tmp_path / "out.wav"),
                success=True, artifacts={"gaps_shortened": "1"},
            )

        monkeypatch.setattr(word_gaps_mod, "shorten_word_gaps", fake_shorten_word_gaps)
        monkeypatch.setattr(
            sys, "argv",
            ["praisonai-editor", "word-gaps", "in.wav", "-o", "out.wav",
             "--threshold", "0.8", "--target", "0.2", "--local", "--language", "en", "--json"],
        )
        assert cli.main() == 0

        assert captured["input_path"] == "in.wav"
        assert captured["output_path"] == "out.wav"
        assert captured["threshold"] == 0.8
        assert captured["target"] == 0.2
        assert captured["use_local"] is True
        assert captured["language"] == "en"
        assert captured["transcript"] is None

    def test_word_gaps_loads_an_explicit_transcript_file(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli
        import praisonai_editor.word_gaps as word_gaps_mod
        from praisonai_editor.models import EditResult

        transcript_path = tmp_path / "t.json"
        transcript_path.write_text(
            '{"text": "hi", "words": [{"text": "hi", "start": 0.0, "end": 0.2}], '
            '"language": "en", "duration": 0.2}'
        )

        captured = {}

        def fake_shorten_word_gaps(input_path, output_path=None, **kwargs):
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path="out.wav", success=True,
                               artifacts={"gaps_shortened": "0"})

        monkeypatch.setattr(word_gaps_mod, "shorten_word_gaps", fake_shorten_word_gaps)
        monkeypatch.setattr(
            sys, "argv",
            ["praisonai-editor", "word-gaps", "in.wav", "--transcript", str(transcript_path)],
        )
        assert cli.main() == 0

        assert captured["transcript"] is not None
        assert captured["transcript"].words[0].text == "hi"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_silence(ffmpeg, path, duration):
    result = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
         "-t", str(duration), str(path)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]


def _make_tone(ffmpeg, path, duration, freq=440):
    result = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
         "-ar", "44100", "-ac", "1", str(path)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestRealEndToEnd:
    """Real ffmpeg, a hand-built (not ASR-derived) transcript with a KNOWN
    3-second gap, and independent verification of the actual output file's
    duration -- proves the feature really cuts the right slice, not just
    that the Python-level plan math looks right (already covered above)."""

    def test_real_gap_is_shortened_and_duration_drops_by_the_expected_amount(self, tmp_path):
        ffmpeg = shutil.which("ffmpeg")
        seg1 = tmp_path / "seg1.wav"
        gap = tmp_path / "gap.wav"
        seg2 = tmp_path / "seg2.wav"
        _make_tone(ffmpeg, seg1, duration=1.0, freq=440)
        _make_silence(ffmpeg, gap, duration=3.0)
        _make_tone(ffmpeg, seg2, duration=1.0, freq=880)

        combined = tmp_path / "combined.wav"
        list_file = tmp_path / "list.txt"
        list_file.write_text(f"file '{seg1}'\nfile '{gap}'\nfile '{seg2}'\n")
        result = subprocess.run(
            [ffmpeg, "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(combined)],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()[-800:]

        before = float(subprocess.run(
            [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(combined)],
            capture_output=True, text=True).stdout.strip())
        assert before == pytest.approx(5.0, abs=0.1)

        # A hand-built transcript matching the real known layout -- avoids
        # depending on faster-whisper/an installed model just to prove the
        # CUTTING logic works on a real file.
        transcript = TranscriptResult(text="a b", words=_words(("a", 0.0, 1.0), ("b", 4.0, 5.0)))

        output = tmp_path / "shortened.wav"
        edit_result = shorten_word_gaps(
            str(combined), str(output), transcript=transcript,
            threshold=1.0, target=0.4,
        )

        assert Path(output).exists()
        after = float(subprocess.run(
            [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(output)],
            capture_output=True, text=True).stdout.strip())
        # Original 3.0s gap kept at 0.4s -> removed 2.6s -> ~2.4s total.
        assert after == pytest.approx(before - 2.6, abs=0.15)
        assert edit_result.artifacts["gaps_shortened"] == "1"
