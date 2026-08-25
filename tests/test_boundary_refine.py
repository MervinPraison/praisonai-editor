"""Tests for praisonai_editor.boundary_refine.

The core claim this pins down: given an ASR-reported cut boundary that's
slightly off (landing in real content instead of the true quiet gap
nearby -- exactly what happens with fast/connected speech), the refined
boundary lands in the REAL gap instead, and never crosses into a
neighboring word's own timestamp no matter how far off the search window
would otherwise push it.
"""

from __future__ import annotations

import numpy as np
import pytest

from praisonai_editor.boundary_refine import (
    _rms_envelope,
    _snap_to_local_min,
    _snap_to_zero_crossing,
    refine_range_boundaries,
)
from praisonai_editor.models import Word


class TestRmsEnvelope:
    def test_loud_frame_has_higher_rms_than_silent_frame(self):
        sr = 1000
        win = 100  # 100ms frames at this sr
        loud = np.full(win, 1.0)
        quiet = np.zeros(win)
        y = np.concatenate([loud, quiet, loud])
        frames = _rms_envelope(y, win)
        assert len(frames) == 3
        assert frames[1] < frames[0]
        assert frames[1] < frames[2]
        assert frames[1] == pytest.approx(0.0)

    def test_drops_a_trailing_partial_frame(self):
        win = 100
        y = np.ones(250)  # 2 full frames + a 50-sample remainder
        frames = _rms_envelope(y, win)
        assert len(frames) == 2


class TestSnapToLocalMin:
    def test_finds_the_real_dip_within_the_search_window(self):
        win = 10
        # Frame index 5 is the quiet one; everything else is loud.
        frames = [1.0] * 10
        frames[5] = 0.01
        result = _snap_to_local_min(frames, win, center_sample=6 * win,
                                     search_samples=5 * win, prefer_earlier=True)
        assert result == 5 * win

    def test_prefer_earlier_breaks_ties_toward_the_first_match(self):
        win = 10
        frames = [1.0, 0.0, 0.0, 0.0, 1.0]
        result = _snap_to_local_min(frames, win, center_sample=2 * win,
                                     search_samples=3 * win, prefer_earlier=True)
        assert result == 1 * win

    def test_prefer_later_breaks_ties_toward_the_last_match(self):
        win = 10
        frames = [1.0, 0.0, 0.0, 0.0, 1.0]
        result = _snap_to_local_min(frames, win, center_sample=2 * win,
                                     search_samples=3 * win, prefer_earlier=False)
        assert result == 3 * win

    def test_empty_frames_returns_center_unchanged(self):
        assert _snap_to_local_min([], 10, center_sample=50, search_samples=30,
                                   prefer_earlier=True) == 50

    def test_out_of_range_center_returns_center_unchanged(self):
        frames = [1.0, 1.0]
        # center_sample way past the end of the frame list, with a search
        # window that still doesn't overlap any real frame.
        result = _snap_to_local_min(frames, 10, center_sample=10_000,
                                     search_samples=5, prefer_earlier=True)
        assert result == 10_000


class TestSnapToZeroCrossing:
    def test_finds_a_real_sign_change_nearby(self):
        y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        # The crossing is between index 2 and 3.
        result = _snap_to_zero_crossing(y, sample_idx=0, search_samples=5)
        assert result in (2, 3)

    def test_no_crossing_in_range_returns_original_index(self):
        y = np.ones(10)  # never changes sign
        result = _snap_to_zero_crossing(y, sample_idx=5, search_samples=2)
        assert result == 5


class TestRefineRangeBoundaries:
    """Real librosa.load round-trip, real constructed WAV files -- not
    mocked. Each test builds raw PCM directly (numpy + soundfile-free via
    the wave module) so the exact sample content is fully controlled."""

    def _write_wav(self, path, samples: np.ndarray, sr: int):
        import wave as wave_mod
        pcm16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        with wave_mod.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm16.tobytes())

    def test_no_ranges_or_no_words_returns_input_unchanged(self, tmp_path):
        assert refine_range_boundaries("whatever.wav", [], [Word(text="a", start=0, end=1)]) == []
        assert refine_range_boundaries("whatever.wav", [(0.0, 1.0)], []) == [(0.0, 1.0)]

    def test_boundary_relocates_into_a_real_nearby_gap_instead_of_clipping_content(self, tmp_path):
        """The central claim: word1 is loud right up to 1.0s, there's a
        genuine ~20ms near-silent gap at [0.99, 1.01)s, then word2 starts
        immediately and is loud again. The ASR reports word1 ending at
        1.05s -- 40ms LATE, deep into word2's real content (exactly the
        "fast/connected speech" failure mode: the ASR boundary doesn't
        match the true acoustic gap). Refinement must relocate the end of
        the removed range back into the real gap, not leave it clipping
        word2."""
        sr = 16000
        loud = np.sin(2 * np.pi * 300 * np.arange(int(0.99 * sr)) / sr) * 0.8
        gap = np.zeros(int(0.02 * sr))          # true silence, 0.99 - 1.01s
        loud2 = np.sin(2 * np.pi * 300 * np.arange(int(0.99 * sr)) / sr) * 0.8
        y = np.concatenate([loud, gap, loud2])
        path = tmp_path / "connected.wav"
        self._write_wav(path, y, sr)

        words = [
            Word(text="word1", start=0.0, end=1.05),   # ASR: 40ms into the real gap+word2
            Word(text="word2", start=1.05, end=2.0),
        ]
        # Removing word1 entirely -- its reported end (1.05) is the range's end.
        refined = refine_range_boundaries(str(path), [(0.0, 1.05)], words)

        assert len(refined) == 1
        _, refined_end = refined[0]
        # The real gap is [0.99, 1.01) -- the refined end must land inside
        # (or right at the edge of) that real quiet window, not at the
        # ASR's own 1.05s guess deep in word2's content.
        assert 0.985 <= refined_end <= 1.015, refined_end

    def test_refinement_never_crosses_into_the_next_words_own_timestamp(self, tmp_path):
        """Pathological case: NO real gap exists at all (words genuinely
        run together with zero silence) -- the energy-minimum search has
        nothing good to find, so the hard clamp against the neighboring
        word's own ASR timestamp is what has to prevent overreach, not
        the energy search itself."""
        sr = 16000
        # Continuously loud from start to end -- no dip anywhere.
        y = np.sin(2 * np.pi * 300 * np.arange(2 * sr) / sr) * 0.8
        path = tmp_path / "no_gap.wav"
        self._write_wav(path, y, sr)

        words = [
            Word(text="word1", start=0.0, end=1.0),
            Word(text="word2", start=1.0, end=2.0),  # touches word1 exactly, no gap
        ]
        refined = refine_range_boundaries(str(path), [(0.0, 1.0)], words, search_ms=200)

        _, refined_end = refined[0]
        # However the energy search resolved, it must never have moved
        # PAST word2's own start (1.0s) -- that would mean eating into
        # word2's real content, exactly what this whole feature exists to
        # prevent.
        assert refined_end <= 1.0 + 1e-6

    def test_refinement_never_crosses_into_the_previous_words_own_timestamp(self, tmp_path):
        sr = 16000
        y = np.sin(2 * np.pi * 300 * np.arange(2 * sr) / sr) * 0.8
        path = tmp_path / "no_gap2.wav"
        self._write_wav(path, y, sr)

        words = [
            Word(text="word1", start=0.0, end=1.0),
            Word(text="word2", start=1.0, end=2.0),
        ]
        # Removing word2 this time -- its start must never refine earlier
        # than word1's own end.
        refined = refine_range_boundaries(str(path), [(1.0, 2.0)], words, search_ms=200)

        refined_start, _ = refined[0]
        assert refined_start >= 1.0 - 1e-6

    def test_multiple_ranges_each_refined_independently(self, tmp_path):
        sr = 16000
        y = np.sin(2 * np.pi * 300 * np.arange(3 * sr) / sr) * 0.8
        # Real gaps at ~1.0s and ~2.0s.
        gap_len = int(0.02 * sr)
        y[int(0.99 * sr):int(0.99 * sr) + gap_len] = 0.0
        y[int(1.99 * sr):int(1.99 * sr) + gap_len] = 0.0
        path = tmp_path / "multi.wav"
        self._write_wav(path, y, sr)

        words = [
            Word(text="a", start=0.0, end=1.05),
            Word(text="b", start=1.05, end=2.05),
            Word(text="c", start=2.05, end=3.0),
        ]
        refined = refine_range_boundaries(str(path), [(0.0, 1.05), (1.05, 2.05)], words)
        assert len(refined) == 2
        # Each range's own end/start should have moved toward its own
        # nearby real gap, independently of the other range.
        assert 0.985 <= refined[0][1] <= 1.015
        assert 1.985 <= refined[1][1] <= 2.015
