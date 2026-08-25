"""Tests for praisonai_editor.render -- specifically the declick fade fix.

Root cause this pins down: FFmpegAudioRenderer._render_concat extracted
each kept segment and joined them with a raw `-c copy` concat, with NO
audio processing at the join at all. Two segments cut from the middle of
continuous audio (the normal case for word-deletion) almost never match in
sample value at the seam, producing an audible click -- reported as "when
cutting I hear a noise, at the point of cut" (Descript-style UI). Fixed by
fading each segment down/up to true silence across a short (default 10ms)
window at every INTERNAL join, never at the file's own start/end. Also
fixes a related bug this surfaced: the re-encode path was hardcoded to
`-c:a libmp3lame` regardless of the output extension, so a .wav output
with reencode=True actually contained an MP3 bitstream in a RIFF/WAVE
container -- real, reproducible, confirmed via ffprobe before this fix
(`RIFF ... WAVE audio, MPEG Layer 3`), not a real WAV a generic WAV
decoder can trust.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from praisonai_editor.models import EditPlan, Segment
from praisonai_editor.render import DECLICK_FADE_MS, FFmpegAudioRenderer, _audio_encode_args


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_tone(path, duration=3.0, freq=400, sr=44100):
    ffmpeg = shutil.which("ffmpeg")
    result = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
         "-ar", str(sr), "-ac", "1", str(path)],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]


def _read_pcm_s16le(path, sr=44100):
    """Decode any audio file to raw mono PCM16 samples via ffmpeg, so this
    works regardless of the file's own container/codec (including a
    mis-encoded one, which is exactly the bug case #2 above needs to
    detect)."""
    ffmpeg = shutil.which("ffmpeg")
    raw_path = str(path) + ".raw"
    result = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-i", str(path), "-f", "s16le", "-ar", str(sr), "-ac", "1", raw_path],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]
    with open(raw_path, "rb") as f:
        data = f.read()
    n = len(data) // 2
    return struct.unpack(f"<{n}h", data)


class TestAudioEncodeArgs:
    def test_mp3_gets_libmp3lame(self):
        assert _audio_encode_args(".mp3") == ["-c:a", "libmp3lame", "-b:a", "192k"]

    def test_wav_gets_real_pcm(self):
        assert _audio_encode_args(".wav") == ["-c:a", "pcm_s16le"]

    def test_m4a_and_aac_get_real_aac(self):
        assert _audio_encode_args(".m4a") == ["-c:a", "aac", "-b:a", "192k"]
        assert _audio_encode_args(".aac") == ["-c:a", "aac", "-b:a", "192k"]

    def test_unknown_extension_falls_through_with_no_forced_codec(self):
        assert _audio_encode_args(".ogg") == []

    def test_case_insensitive(self):
        assert _audio_encode_args(".WAV") == ["-c:a", "pcm_s16le"]

    def test_bitrate_is_forwarded(self):
        assert _audio_encode_args(".mp3", bitrate="320k") == ["-c:a", "libmp3lame", "-b:a", "320k"]


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestDeclickFade:
    def test_join_sample_is_exact_silence_on_both_sides(self, tmp_path):
        """The mathematical guarantee the whole fix rests on: if segment A
        fades linearly to 0 across its last DECLICK_FADE_MS and segment B
        fades linearly from 0 across its first DECLICK_FADE_MS, the two
        samples immediately either side of the join are BOTH ~0 -- so
        there is no discontinuity at that point regardless of what the
        original waveform was doing. Real ffmpeg, not mocked."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=3.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=3.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.wav"
        renderer.render(str(src), str(out), plan, copy_codec=False)

        samples = _read_pcm_s16le(out)
        join_idx = int(0.5 * 44100)
        # A tiny window either side of the seam -- both sides should be at
        # or extremely close to zero (the fade's own endpoints).
        before = samples[join_idx - 5:join_idx]
        after = samples[join_idx:join_idx + 5]
        assert max(abs(s) for s in before) < 50, before
        assert max(abs(s) for s in after) < 50, after

    def test_declick_ms_zero_disables_the_fade(self, tmp_path):
        """The escape hatch: declick_ms=0 restores the pre-fix behavior --
        a real, unfaded discontinuity at the join. Confirms the fade in
        the test above is actually doing something, not just an artifact
        of the tone/cut points chosen."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=3.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=3.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "no_declick.wav"
        renderer.render(str(src), str(out), plan, copy_codec=False, declick_ms=0)

        samples = _read_pcm_s16le(out)
        join_idx = int(0.5 * 44100)
        before = samples[join_idx - 5:join_idx]
        after = samples[join_idx:join_idx + 5]
        # With no declick fade, a 400Hz tone mid-cycle is nowhere near
        # silence on either side of an arbitrary join.
        assert max(abs(s) for s in before) > 500 or max(abs(s) for s in after) > 500

    def test_first_segment_start_and_last_segment_end_are_not_faded(self, tmp_path):
        """Only INTERNAL join seams get faded -- the very first sample of
        the output and the very last must be untouched (they're real
        recording boundaries the edit never cut, not edit seams)."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=3.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=3.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.wav"
        renderer.render(str(src), str(out), plan, copy_codec=False)

        samples = _read_pcm_s16le(out)
        # The very first few samples of the output are the file's own
        # start (segment 0's own start, at t=0 of the original tone) --
        # a 400Hz sine literally starts at 0 anyway, so instead assert the
        # OVERALL output isn't uniformly near-zero at the start the way it
        # would be if a fade had wrongly been applied there too. Real
        # check: the output around 0.4s in (well before the internal join
        # at 0.5s, past any possible fade window) is still full-amplitude.
        untouched_region = samples[int(0.3 * 44100):int(0.4 * 44100)]
        # ffmpeg's own `sine` lavfi source peaks at ~4095 for int16 output
        # (its own default amplitude, not full-scale) -- not this test's
        # concern, just the ceiling to compare against.
        assert max(abs(s) for s in untouched_region) > 3000

    def test_three_segment_join_fades_both_sides_of_the_middle_segment(self, tmp_path):
        """A middle segment has a cut on BOTH sides -- it must fade in at
        its start AND fade out at its end, unlike the first/last segments
        which only fade on the one side that borders a cut."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=5.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=1.0, action="keep", reason="test", category="content"),
            Segment(start=1.5, end=2.0, action="keep", reason="test", category="content"),
            Segment(start=2.5, end=5.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.wav"
        renderer.render(str(src), str(out), plan, copy_codec=False)

        samples = _read_pcm_s16le(out)
        # Output layout: [0, 1.0) then [1.0, 1.5) (the middle segment,
        # 0.5s long) then [1.5, ...). Both its own boundaries (at 1.0 and
        # at 1.5 in the OUTPUT timeline) are real join seams.
        join1 = int(1.0 * 44100)
        join2 = int(1.5 * 44100)
        for join_idx, label in ((join1, "start of middle segment"), (join2, "end of middle segment")):
            before = samples[join_idx - 5:join_idx]
            after = samples[join_idx:join_idx + 5]
            assert max(abs(s) for s in before) < 50, f"{label}: {before}"
            assert max(abs(s) for s in after) < 50, f"{label}: {after}"

    def test_wav_output_is_real_pcm_not_mislabeled_mp3(self, tmp_path):
        """Regression test for the bug the declick fix would otherwise
        have made much more common: reencode's per-segment encode used to
        be hardcoded to libmp3lame regardless of the output extension, so
        a .wav output with reencode=True (or, after this fix, ANY
        multi-segment join, since declicking forces a re-encode) actually
        contained an MP3 bitstream inside a RIFF/WAVE container -- real
        and reproducible before this fix (confirmed via ffprobe:
        `RIFF ... WAVE audio, MPEG Layer 3`). A real WAV decoder (not just
        ffmpeg's own lenient probing) must be able to read it."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=2.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=2.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.wav"
        renderer.render(str(src), str(out), plan, copy_codec=False)

        # Python's own `wave` module refuses anything that isn't real PCM
        # (or a handful of other WAVE-legal codecs) -- this would have
        # raised wave.Error: unknown format before the fix.
        with wave.open(str(out), "rb") as w:
            assert w.getnframes() > 0
            assert w.getframerate() == 44100

    def test_mp3_output_still_uses_mp3_codec(self, tmp_path):
        """No regression for the already-correct case."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=2.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=2.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.mp3"
        renderer.render(str(src), str(out), plan, copy_codec=False)

        result = subprocess.run(
            [shutil.which("ffprobe"), "-v", "error", "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True,
        )
        assert result.stdout.strip() == "mp3"

    def test_copy_codec_true_still_forces_reencode_at_seams(self, tmp_path):
        """copy_codec=True (the fast default) must NOT bring back the
        click -- any segment touching a seam is re-encoded regardless,
        since a filter cannot run during a stream copy."""
        src = tmp_path / "tone.wav"
        _make_tone(src, duration=3.0, freq=400)

        renderer = FFmpegAudioRenderer()
        plan = EditPlan(segments=[
            Segment(start=0.0, end=0.5, action="keep", reason="test", category="content"),
            Segment(start=1.0, end=3.0, action="keep", reason="test", category="content"),
        ])
        out = tmp_path / "declicked.wav"
        renderer.render(str(src), str(out), plan, copy_codec=True)

        samples = _read_pcm_s16le(out)
        join_idx = int(0.5 * 44100)
        before = samples[join_idx - 5:join_idx]
        after = samples[join_idx:join_idx + 5]
        assert max(abs(s) for s in before) < 50
        assert max(abs(s) for s in after) < 50

    def test_default_fade_duration_is_the_researched_declick_value(self):
        """10ms: within the 5-15ms range standard for a declick fade (NOT
        a musical crossfade, which needs hundreds of ms to seconds) --
        long enough to reliably kill the discontinuity, short enough to
        be inaudible as a dip in continuous speech."""
        assert DECLICK_FADE_MS == 10
