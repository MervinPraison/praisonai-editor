"""Tests for manual time-range removal."""

import shutil
import subprocess
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import praisonai_editor.remove_ranges as remove_ranges_mod
from praisonai_editor.models import ProbeResult, TranscriptResult, Word
from praisonai_editor.remove_ranges import (
    _retime_transcript,
    build_remove_plan,
    parse_time,
    parse_time_range,
    remove_time_ranges,
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_wav(path, samples, sr):
    pcm16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


class TestParseTime:
    def test_seconds(self):
        assert parse_time("713") == 713.0
        assert parse_time(713.5) == 713.5

    def test_mm_ss(self):
        assert parse_time("11:53") == 11 * 60 + 53

    def test_hh_mm_ss(self):
        assert parse_time("1:11:53") == 3600 + 11 * 60 + 53


class TestParseTimeRange:
    def test_dash(self):
        assert parse_time_range("11:53-12:43") == (713.0, 763.0)

    def test_comma(self):
        assert parse_time_range("11:53,12:43") == (713.0, 763.0)

    def test_tuple(self):
        assert parse_time_range(("11:53", "12:43")) == (713.0, 763.0)

    def test_end_before_start(self):
        with pytest.raises(ValueError):
            parse_time_range("12:43-11:53")


class TestBuildRemovePlan:
    def test_single_middle_cut(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        keep = plan.get_keep_segments()
        assert len(keep) == 2
        assert keep[0].start == 0.0 and keep[0].end == 20.0
        assert keep[1].start == 30.0 and keep[1].end == 100.0
        assert plan.removed_duration == pytest.approx(10.0)
        assert plan.edited_duration == pytest.approx(90.0)

    def test_overlapping_ranges_merge(self):
        plan = build_remove_plan(100.0, [(10.0, 25.0), (20.0, 40.0)])
        assert plan.removed_duration == pytest.approx(30.0)

    def test_range_beyond_duration(self):
        with pytest.raises(ValueError):
            build_remove_plan(60.0, [(0.0, 61.0)])


class TestRetimeTranscript:
    """_retime_transcript walks plan.segments (the already-merged KEPT
    segments build_remove_plan produces), not a re-derived merge of the raw
    remove_ranges input -- these tests prove that by feeding overlapping
    input ranges through build_remove_plan first, the same way
    remove_time_ranges itself does."""

    def test_word_entirely_before_a_cut_is_unchanged(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        words = [Word(text="a", start=5.0, end=6.0, confidence=0.9)]
        transcript = TranscriptResult(text="a", words=words, language="en", duration=100.0)

        retimed = _retime_transcript(transcript, plan)

        assert len(retimed.words) == 1
        assert retimed.words[0].start == pytest.approx(5.0)
        assert retimed.words[0].end == pytest.approx(6.0)
        assert retimed.words[0].confidence == pytest.approx(0.9)

    def test_word_entirely_after_a_single_cut_shifts_by_that_cuts_duration(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])  # 10s removed
        words = [Word(text="b", start=40.0, end=41.0)]
        transcript = TranscriptResult(text="b", words=words)

        retimed = _retime_transcript(transcript, plan)

        assert retimed.words[0].start == pytest.approx(30.0)
        assert retimed.words[0].end == pytest.approx(31.0)

    def test_word_between_two_cuts_shifts_by_only_the_first_cuts_duration(self):
        # Cuts at 10-15 (5s) and 40-50 (10s); a word at 20-21 sits between
        # them -- summing BOTH cuts' durations (15s) would be wrong, only
        # the first (5s) has actually happened "before" this word yet.
        plan = build_remove_plan(100.0, [(10.0, 15.0), (40.0, 50.0)])
        words = [Word(text="c", start=20.0, end=21.0)]
        transcript = TranscriptResult(text="c", words=words)

        retimed = _retime_transcript(transcript, plan)

        assert retimed.words[0].start == pytest.approx(15.0)
        assert retimed.words[0].end == pytest.approx(16.0)

    def test_word_straddling_a_cut_boundary_is_dropped(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        words = [Word(text="d", start=19.0, end=21.0)]  # straddles the cut start
        transcript = TranscriptResult(text="d", words=words)

        retimed = _retime_transcript(transcript, plan)

        assert retimed.words == []

    def test_word_entirely_inside_the_removed_range_is_dropped(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        words = [Word(text="e", start=22.0, end=25.0)]
        transcript = TranscriptResult(text="e", words=words)

        retimed = _retime_transcript(transcript, plan)

        assert retimed.words == []

    def test_overlapping_input_ranges_merged_by_build_remove_plan_still_shift_correctly(self):
        # (10-25) and (20-40) overlap and build_remove_plan merges them into
        # a single removed range 10-40 (30s total) BEFORE _retime_transcript
        # ever sees plan.segments -- proving this uses plan.segments rather
        # than re-deriving its own (possibly-disagreeing) merge.
        plan = build_remove_plan(100.0, [(10.0, 25.0), (20.0, 40.0)])
        words = [
            Word(text="before", start=5.0, end=6.0),
            Word(text="inside", start=22.0, end=23.0),  # inside the merged gap
            Word(text="after", start=50.0, end=51.0),
        ]
        transcript = TranscriptResult(text="before inside after", words=words)

        retimed = _retime_transcript(transcript, plan)

        kept = {w.text: w for w in retimed.words}
        assert set(kept) == {"before", "after"}
        assert kept["before"].start == pytest.approx(5.0)
        # 30s removed total (the merged range), not e.g. 15 + 20 = 35s from
        # naively re-merging the raw, still-overlapping input ranges.
        assert kept["after"].start == pytest.approx(20.0)
        assert kept["after"].end == pytest.approx(21.0)

    def test_retimed_result_carries_text_language_and_edited_duration(self):
        plan = build_remove_plan(100.0, [(20.0, 30.0)])
        words = [Word(text="hello", start=5.0, end=6.0), Word(text="world", start=40.0, end=41.0)]
        transcript = TranscriptResult(text="hello world", words=words, language="fr", duration=100.0)

        retimed = _retime_transcript(transcript, plan)

        assert retimed.text == "hello world"
        assert retimed.language == "fr"
        assert retimed.duration == pytest.approx(plan.edited_duration)


class TestRemoveTimeRangesTranscriptParam:
    """remove_time_ranges' own transcript= kwarg -- ffmpeg itself is
    monkeypatched out (subprocess.run + _find_ffmpeg), the same convention
    tests/test_conform.py already uses for a pure-logic-focused test of a
    function that otherwise shells out."""

    def _mock_ffmpeg(self, monkeypatch):
        calls = []

        def _fake_run(cmd, *args, **kwargs):
            calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        import praisonai_editor.render as render_mod
        monkeypatch.setattr(render_mod, "_find_ffmpeg", lambda: "ffmpeg")
        return calls

    def _mock_probe(self, monkeypatch, duration):
        monkeypatch.setattr(
            remove_ranges_mod, "probe_media",
            lambda path: ProbeResult(path=str(path), duration=duration, has_video=False),
        )

    def test_without_transcript_kwarg_result_transcript_stays_none(self, tmp_path, monkeypatch):
        """No behavior change for every existing caller that doesn't pass
        transcript= -- e.g. pipeline.py's own use of remove_time_ranges (if
        any) or the Studio worker's pre-sidecar call sites."""
        self._mock_ffmpeg(monkeypatch)
        self._mock_probe(monkeypatch, 100.0)
        src = tmp_path / "in.wav"
        src.write_bytes(b"\x00")

        result = remove_time_ranges(
            str(src), [(20.0, 30.0)], output_path=str(tmp_path / "out.wav"))

        assert result.success
        assert result.transcript is None

    def test_with_transcript_kwarg_result_transcript_is_retimed(self, tmp_path, monkeypatch):
        self._mock_ffmpeg(monkeypatch)
        self._mock_probe(monkeypatch, 100.0)
        src = tmp_path / "in.wav"
        src.write_bytes(b"\x00")
        words = [
            Word(text="before", start=5.0, end=6.0),
            Word(text="after", start=40.0, end=41.0),
        ]
        transcript = TranscriptResult(text="before after", words=words, language="en", duration=100.0)

        result = remove_time_ranges(
            str(src), [(20.0, 30.0)], output_path=str(tmp_path / "out.wav"),
            transcript=transcript)

        assert result.success
        assert result.transcript is not None
        kept = {w.text: w for w in result.transcript.words}
        assert kept["before"].start == pytest.approx(5.0)
        assert kept["after"].start == pytest.approx(30.0)
        assert result.transcript.duration == pytest.approx(90.0)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestRemoveTimeRangesBoundaryRefinement:
    """Real ffmpeg, real constructed audio -- the exact "fast/connected
    speech" failure mode a real ASR would produce: a word's reported end
    timestamp lands inside the NEXT word's real content instead of the
    true (short) silent gap between them. Proves remove_time_ranges'
    default refine_boundaries=True actually relocates the cut, and that
    turning it off restores the old (clipping) behavior -- an escape
    hatch, not a silent behavior change with no way back."""

    def _make_connected_speech(self, path):
        sr = 16000
        # word1: loud tone for 0.99s. A real ~20ms silent gap. word2:
        # loud tone starting immediately after. The transcript below
        # deliberately reports word1 ending 40ms INTO word2's real
        # content (0.99 + 0.02 + 0.02 = 1.05s), not at the true boundary.
        word1 = np.sin(2 * np.pi * 300 * np.arange(int(0.99 * sr)) / sr) * 0.8
        gap = np.zeros(int(0.02 * sr))
        word2 = np.sin(2 * np.pi * 440 * np.arange(int(1.0 * sr)) / sr) * 0.8
        y = np.concatenate([word1, gap, word2])
        _write_wav(path, y, sr)
        return sr

    def test_refine_boundaries_true_avoids_clipping_the_next_words_onset(self, tmp_path):
        src = tmp_path / "connected.wav"
        self._make_connected_speech(src)
        words = [
            Word(text="word1", start=0.0, end=1.05),
            Word(text="word2", start=1.05, end=2.0),
        ]
        transcript = TranscriptResult(text="word1 word2", words=words, language="en", duration=2.0)

        result = remove_time_ranges(
            str(src), [(0.0, 1.05)], output_path=str(tmp_path / "refined.wav"),
            reencode=True, transcript=transcript, refine_boundaries=True,
        )

        assert result.success
        # word2 in the retimed transcript must start with a small REAL
        # gap of silence before it (the refined cut landed in the true
        # ~20ms gap, not 40ms into word2 itself) -- not at 0.0, which
        # would mean word2's own onset got clipped off along with word1.
        retimed_word2 = result.transcript.words[0]
        assert retimed_word2.text == "word2"
        assert retimed_word2.start > 0.02, (
            f"word2 starts at {retimed_word2.start}s in the output -- "
            "too close to 0 to have avoided clipping its own onset"
        )

    def test_refine_boundaries_false_restores_the_old_clipping_behavior(self, tmp_path):
        """The escape hatch: explicitly opting out must reproduce exactly
        what remove_time_ranges did before this feature existed -- cut at
        the raw reported timestamp, onset-clipping and all."""
        src = tmp_path / "connected.wav"
        self._make_connected_speech(src)
        words = [
            Word(text="word1", start=0.0, end=1.05),
            Word(text="word2", start=1.05, end=2.0),
        ]
        transcript = TranscriptResult(text="word1 word2", words=words, language="en", duration=2.0)

        result = remove_time_ranges(
            str(src), [(0.0, 1.05)], output_path=str(tmp_path / "raw.wav"),
            reencode=True, transcript=transcript, refine_boundaries=False,
        )

        retimed_word2 = result.transcript.words[0]
        # Cutting at the raw 1.05s boundary removes word1's full 1.05s,
        # so word2 (which starts at 1.05s in the ORIGINAL) lands at
        # exactly 0.0s in the output -- its own onset already clipped by
        # the 40ms the raw ASR timestamp overshot by.
        assert retimed_word2.start == pytest.approx(0.0, abs=0.01)

    def test_default_is_refine_boundaries_true(self, tmp_path):
        """No caller changes needed -- every existing remove_ranges/
        word_gaps call with a transcript benefits immediately."""
        src = tmp_path / "connected.wav"
        self._make_connected_speech(src)
        words = [
            Word(text="word1", start=0.0, end=1.05),
            Word(text="word2", start=1.05, end=2.0),
        ]
        transcript = TranscriptResult(text="word1 word2", words=words, language="en", duration=2.0)

        result = remove_time_ranges(
            str(src), [(0.0, 1.05)], output_path=str(tmp_path / "default.wav"),
            reencode=True, transcript=transcript,
        )

        assert result.transcript.words[0].start > 0.02

    def test_no_transcript_means_no_refinement_attempted(self, tmp_path, monkeypatch):
        """refine_boundaries has nothing to clamp/search against without a
        transcript's word list -- must not even try (and definitely must
        not error) when transcript=None, the majority of real calls."""
        src = tmp_path / "connected.wav"
        self._make_connected_speech(src)

        called = {"n": 0}
        import praisonai_editor.boundary_refine as boundary_refine_mod
        real_refine = boundary_refine_mod.refine_range_boundaries

        def spy(*args, **kwargs):
            called["n"] += 1
            return real_refine(*args, **kwargs)

        monkeypatch.setattr(boundary_refine_mod, "refine_range_boundaries", spy)

        result = remove_time_ranges(
            str(src), [(0.0, 1.05)], output_path=str(tmp_path / "no_transcript.wav"),
            reencode=True,
        )

        assert result.success
        assert called["n"] == 0


class TestRemoveCliArgWiring:
    """No CLI-level test existed for `remove` at all before this -- these
    cover the two new flags (--transcript, --no-refine-boundaries) plus a
    no-regression pin for the existing flags."""

    def test_transcript_and_refine_boundaries_default_wiring(self, monkeypatch, tmp_path):
        import sys as sys_mod

        import praisonai_editor.cli as cli
        import praisonai_editor.remove_ranges as remove_ranges_mod
        from praisonai_editor.models import EditResult

        transcript_path = tmp_path / "t.json"
        transcript_path.write_text(
            '{"text": "hi", "words": [{"text": "hi", "start": 0.0, "end": 0.2}], '
            '"language": "en", "duration": 0.2}'
        )

        captured = {}

        def fake_remove_time_ranges(input_path, ranges, **kwargs):
            captured["input_path"] = input_path
            captured["ranges"] = ranges
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path="out.wav", success=True)

        monkeypatch.setattr(remove_ranges_mod, "remove_time_ranges", fake_remove_time_ranges)
        monkeypatch.setattr(
            sys_mod, "argv",
            ["praisonai-editor", "remove", "in.wav", "--range", "1.0-2.0",
             "--transcript", str(transcript_path), "--json"],
        )
        assert cli.main() == 0

        assert captured["ranges"] == ["1.0-2.0"]
        assert captured["transcript"] is not None
        assert captured["transcript"].words[0].text == "hi"
        # Default: refine_boundaries stays on unless --no-refine-boundaries is passed.
        assert captured["refine_boundaries"] is True

    def test_no_refine_boundaries_flag_disables_it(self, monkeypatch, tmp_path):
        import sys as sys_mod

        import praisonai_editor.cli as cli
        import praisonai_editor.remove_ranges as remove_ranges_mod
        from praisonai_editor.models import EditResult

        transcript_path = tmp_path / "t.json"
        transcript_path.write_text('{"text": "hi", "words": [], "language": "en", "duration": 0.2}')

        captured = {}

        def fake_remove_time_ranges(input_path, ranges, **kwargs):
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path="out.wav", success=True)

        monkeypatch.setattr(remove_ranges_mod, "remove_time_ranges", fake_remove_time_ranges)
        monkeypatch.setattr(
            sys_mod, "argv",
            ["praisonai-editor", "remove", "in.wav", "--range", "1.0-2.0",
             "--transcript", str(transcript_path), "--no-refine-boundaries", "--json"],
        )
        assert cli.main() == 0
        assert captured["refine_boundaries"] is False

    def test_no_transcript_means_transcript_none(self, monkeypatch):
        import sys as sys_mod

        import praisonai_editor.cli as cli
        import praisonai_editor.remove_ranges as remove_ranges_mod
        from praisonai_editor.models import EditResult

        captured = {}

        def fake_remove_time_ranges(input_path, ranges, **kwargs):
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path="out.wav", success=True)

        monkeypatch.setattr(remove_ranges_mod, "remove_time_ranges", fake_remove_time_ranges)
        monkeypatch.setattr(
            sys_mod, "argv",
            ["praisonai-editor", "remove", "in.wav", "--range", "1.0-2.0", "--json"],
        )
        assert cli.main() == 0
        assert captured["transcript"] is None
