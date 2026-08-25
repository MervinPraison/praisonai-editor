"""Refine ASR-reported word-cut boundaries against the real waveform.

A transcription engine's word-level timestamp is a coarse HINT, not a
precise acoustic boundary -- especially for fast/connected speech where
words run together with no clean gap between them (the ASR model wasn't
optimized for frame-accurate timing; forced-alignment research confirms
even purpose-built aligners only get ~90% of boundaries within 50ms, and
plain ASR-derived timestamps are considerably looser than that). Cutting
exactly at the reported boundary can clip the onset of the kept word on
either side, or leave a trailing fragment of the removed word attached to
a neighbor -- audible as an "abrupt"-sounding edit even when the join
itself has no click (that's a separate problem -- see render.py's own
declick fade, which still applies AFTER this).

This is deliberately NOT a re-alignment model (WhisperX/wav2vec2 forced
alignment, Montreal Forced Aligner, etc.) -- those need an extra heavy
model dependency and, per real-world reports, aren't reliably more
accurate at the sub-boundary level this problem actually lives at. This
is the same lightweight, three-step DSP technique real open-source
word-level audio editors use (e.g. github.com/dougcalobrisi/erm's own
refine.py) and that librosa's own docs recommend for exactly this reason
(onset_detect's `backtrack=True` "backtrack[s] from each peak to a
preceding local minimum of energy... primarily useful when using onsets
as slice points for segmentation"):

  1. Search a small window (default 60ms) around the ASR-reported
     boundary for the REAL local minimum in short-time energy (RMS) --
     the actual quietest instant nearby, which is far more likely to be
     a genuine word gap than the ASR's own guess.
  2. Snap that point to the nearest true zero-crossing, so the refined
     boundary doesn't itself introduce a new click (declick_ms in
     render.py's afade still runs on top of this regardless, as defense
     in depth).
  3. Hard-clamp the whole search so it can NEVER cross into a
     neighboring KEPT word's own ASR timestamp -- the guarantee that
     bounds how "aggressive" this can ever be: at worst, it does nothing
     (falls back to the original boundary), it never eats real content
     from the word on either side.

Usage:
    from praisonai_editor.boundary_refine import refine_range_boundaries

    refined = refine_range_boundaries("interview.wav", [(4.12, 4.58)], transcript.words)
"""

from __future__ import annotations

import subprocess
from typing import List, Sequence, Tuple

from .models import Word
from .render import _find_ffmpeg

RangeSpec = Tuple[float, float]

#: Sample rate everything is analyzed at. Fixed rather than "whatever the
#: source file uses" so the RMS/zero-crossing math below has one known
#: scale -- ffmpeg resamples to this on the way in, same posture as
#: detect.py's own librosa-based analysis (also fixed-rate).
_ANALYSIS_SR = 16000


def _decode_to_mono_float32(input_path: str):
    """Decode `input_path` to mono float32 samples at _ANALYSIS_SR via
    ffmpeg -- deliberately NOT librosa.load: this keeps the whole module's
    only dependency an ffmpeg subprocess call (already required
    everywhere else in this package) rather than pulling in librosa's own
    scipy dependency chain purely to read samples for a lightweight DSP
    pass. Real crash this avoided in practice: a broken scipy.sparse.linalg
    PROPACK binary in one dev environment took down librosa.load() (a
    lazy-loaded, unrelated-looking scipy submodule import triggered deep
    inside librosa's own loader) despite bare `import librosa` working
    fine -- a class of binary-compatibility risk (ARM/x86, Python version,
    wheel mismatches) this module has no reason to inherit just to read
    PCM samples ffmpeg can hand over directly.
    """
    import numpy as np

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-nostdin", "-i", input_path,
        "-f", "f32le", "-ar", str(_ANALYSIS_SR), "-ac", "1",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"FFmpeg decode failed: {stderr[-800:]}")
    y = np.frombuffer(result.stdout, dtype="float32")
    return y, _ANALYSIS_SR


def _rms_envelope(y, win: int) -> List[float]:
    """Short-time RMS energy, one value per `win`-sample frame (the last
    partial frame, if any, is dropped -- too short to be a meaningful
    energy estimate)."""
    import numpy as np

    n_frames = len(y) // win
    frames = []
    for i in range(n_frames):
        chunk = y[i * win:(i + 1) * win]
        frames.append(float(np.sqrt(np.mean(chunk.astype("float64") ** 2))))
    return frames


def _snap_to_local_min(frames: List[float], win: int, center_sample: int,
                        search_samples: int, prefer_earlier: bool) -> int:
    """The sample index (a frame boundary) of the lowest-RMS frame within
    `search_samples` of `center_sample`.

    Ties are broken toward the EARLIEST candidate for a start-edge and the
    LATEST for an end-edge (prefer_earlier controls which) -- both biases
    point the same direction: toward including a little more silence in
    the removed range rather than a little less, since a few extra
    milliseconds of silence removed is inaudible but a fragment left
    behind is not. Returns `center_sample` unchanged if there are no
    frames to search (e.g. a file shorter than one window).
    """
    if not frames:
        return center_sample
    center_frame = center_sample // win
    span_frames = max(1, search_samples // win)
    lo = max(0, center_frame - span_frames)
    hi = min(len(frames) - 1, center_frame + span_frames)
    if lo > hi:
        return center_sample
    window = frames[lo:hi + 1]
    min_val = min(window)
    # Every frame within a hair of the true minimum counts as a tie --
    # floating-point RMS values from real audio essentially never repeat
    # exactly, so a small relative tolerance is what actually lets ties
    # (e.g. a flat stretch of near-silence) resolve consistently.
    tol = min_val * 1e-6 + 1e-9
    matches = [lo + i for i, v in enumerate(window) if v <= min_val + tol]
    chosen_frame = matches[0] if prefer_earlier else matches[-1]
    return chosen_frame * win


def _snap_to_zero_crossing(y, sample_idx: int, search_samples: int) -> int:
    """The nearest true sign-change in `y` to `sample_idx`, within
    `search_samples` either side -- avoids the refined boundary itself
    landing mid-swing. Returns `sample_idx` unchanged if no crossing is
    found in range (e.g. true digital silence, which has no sign changes
    but also can't click)."""
    lo = max(0, sample_idx - search_samples)
    hi = min(len(y) - 2, sample_idx + search_samples)
    best_idx = sample_idx
    best_dist = None
    for i in range(lo, hi + 1):
        if y[i] == 0 or (y[i] > 0) != (y[i + 1] > 0):
            dist = abs(i - sample_idx)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx


def refine_range_boundaries(
    input_path: str,
    ranges: Sequence[RangeSpec],
    words: Sequence[Word],
    *,
    search_ms: float = 60.0,
    win_ms: float = 10.0,
    zero_cross_ms: float = 5.0,
) -> List[RangeSpec]:
    """Nudge each (start, end) in `ranges` to a real acoustic gap nearby.

    Args:
        input_path: The audio file `ranges` and `words` are timestamped
            against (loaded read-only here; the actual cut still happens
            in remove_ranges.py/ffmpeg).
        ranges: Time ranges about to be removed, e.g. `[(word.start, word.end)]`
            for a single deleted word, or one entry per selected word/run.
        words: The FULL transcript word list (time-ordered), used only to
            compute each range's clamp bounds -- the end of whatever kept
            word precedes it, and the start of whatever kept word follows.
        search_ms: How far either side of each reported boundary to search
            for a better (quieter) cut point. 60ms matches the range real
            open-source implementations of this technique use -- long
            enough to find a genuine micro-gap in fast speech, short
            enough that it can never plausibly span a whole neighboring
            word.
        win_ms: RMS analysis frame size. 10ms is standard short-time energy
            resolution -- fine enough to localize a gap, coarse enough to
            average over a single glottal pulse rather than chasing noise.
        zero_cross_ms: How far to look for a zero-crossing after the
            energy-minimum search lands -- a few ms, not the full search
            window, since this step is pure click-safety, not gap-finding.

    Returns:
        A new list of (start, end) tuples, same length and order as
        `ranges`, each refined independently. If `ranges` is empty, or
        `words` is empty (no clamp information available), returns
        `ranges` unchanged rather than guessing.
    """
    if not ranges or not words:
        return list(ranges)

    y, sr = _decode_to_mono_float32(input_path)
    win = max(1, int(sr * win_ms / 1000.0))
    frames = _rms_envelope(y, win)
    search_samples = max(1, int(sr * search_ms / 1000.0))
    zc_samples = max(1, int(sr * zero_cross_ms / 1000.0))
    duration_samples = len(y)

    sorted_words = sorted(words, key=lambda w: w.start)

    refined: List[RangeSpec] = []
    for start, end in ranges:
        # The clamp: how far this range's edges are allowed to move.
        # Start can move earlier, toward (but never past) the end of
        # whatever kept word precedes it -- 0.0 if this range starts at
        # the very beginning of the file. End can move later, toward
        # (but never past) the start of whatever kept word follows -- the
        # file's own duration if this range runs to the end.
        prev_end = max((w.end for w in sorted_words if w.end <= start), default=0.0)
        next_start = min((w.start for w in sorted_words if w.start >= end),
                          default=duration_samples / sr)

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        clamp_lo = int(prev_end * sr)
        clamp_hi = min(duration_samples, int(next_start * sr))

        refined_start = _snap_to_local_min(frames, win, start_sample, search_samples,
                                            prefer_earlier=True)
        refined_start = _snap_to_zero_crossing(y, refined_start, zc_samples)
        refined_start = max(clamp_lo, min(refined_start, end_sample))

        refined_end = _snap_to_local_min(frames, win, end_sample, search_samples,
                                          prefer_earlier=False)
        refined_end = _snap_to_zero_crossing(y, refined_end, zc_samples)
        refined_end = min(clamp_hi, max(refined_end, start_sample))

        if refined_end <= refined_start:
            # A pathological window (e.g. back-to-back words with no gap
            # at all, or a search radius that overshot both ways into the
            # same tiny window) -- fall back to the original, unrefined
            # boundary rather than emit a zero/negative-length range.
            refined.append((start, end))
        else:
            refined.append((refined_start / sr, refined_end / sr))

    return refined
