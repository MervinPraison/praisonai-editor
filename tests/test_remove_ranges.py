"""Tests for manual time-range removal."""

import subprocess
from types import SimpleNamespace

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
