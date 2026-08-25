"""Shorten long pauses between words down to a target length.

A surgical complement to full silence removal (plan.py's detect_silences):
that heuristic deletes a gap entirely, this one keeps a fixed amount of
breathing room after each word and only trims the excess -- natural rhythm
survives, dead air doesn't.

Usage:
    from praisonai_editor.word_gaps import shorten_word_gaps

    result = shorten_word_gaps("interview.wav")                     # auto-transcribes
    result = shorten_word_gaps("interview.wav", transcript=my_transcript,
                                threshold=1.0, target=0.3)
"""

from __future__ import annotations

from typing import List, Tuple

from .models import EditResult, TranscriptResult
from .remove_ranges import remove_time_ranges


def find_long_gaps(transcript: TranscriptResult, threshold: float) -> List[Tuple[float, float]]:
    """The [end-of-word, start-of-next-word) span of every inter-word gap
    longer than `threshold` seconds, in transcript (time) order."""
    words = transcript.words
    gaps = []
    for i in range(len(words) - 1):
        gap_start = words[i].end
        gap_end = words[i + 1].start
        if gap_end - gap_start > threshold:
            gaps.append((gap_start, gap_end))
    return gaps


def shorten_word_gaps(
    input_path: str,
    output_path: str | None = None,
    *,
    transcript: TranscriptResult | None = None,
    use_local: bool = True,
    language: str | None = None,
    model: str | None = None,
    threshold: float = 0.5,
    target: float = 0.25,
    reencode: bool = False,
    verbose: bool = False,
) -> EditResult:
    """Trim any inter-word pause longer than `threshold` seconds down to `target` seconds.

    Built on remove_time_ranges: shortening a gap IS removing the excess
    slice of it, so this is a thin wrapper that computes which slices
    qualify and lets that function do the actual cutting AND the
    server-proven transcript re-timing (its own `transcript=` kwarg).

    Args:
        input_path: Source audio file.
        output_path: Destination path (default: remove_time_ranges' own
            `{stem}_cut{ext}`).
        transcript: Word-level transcript for this exact file. If omitted,
            transcribes it first (use_local/language/model control that
            pass), so this works standalone without a separate transcribe
            step -- the same posture phrase_trim already has.
        use_local, language, model: Only used when `transcript` is None.
        threshold: Only gaps longer than this (seconds) are touched.
        target: What each qualifying gap is shortened TO (seconds). Must be
            less than `threshold` -- otherwise nothing would ever qualify.
        reencode: Re-encode instead of stream-copy (slower, frame-accurate).
        verbose: Print ffmpeg progress.

    Returns:
        EditResult with `.transcript` re-timed to match, and
        `artifacts["gaps_shortened"]`/`["threshold"]`/`["target"]` set.

    Raises:
        ValueError: `target` is negative or >= `threshold`, or no gap in
            the transcript exceeds `threshold` (nothing to shorten).
    """
    if target < 0:
        raise ValueError(f"target must be >= 0, got {target}")
    if target >= threshold:
        raise ValueError(f"target ({target}) must be less than threshold ({threshold})")

    if transcript is None:
        from .transcribe import transcribe_audio

        # vad_filter=True: without it, a long pause is often absorbed into
        # the timestamp of the word right before or after it instead of
        # showing up as a real gap between two words -- exactly the data
        # find_long_gaps needs to be accurate. See LocalTranscriber's own
        # docstring for the full explanation; safe to always request here
        # even for the OpenAI path, where it's simply a no-op.
        transcript = transcribe_audio(
            input_path, use_local=use_local, language=language, model=model, vad_filter=True
        )

    gaps = find_long_gaps(transcript, threshold)
    if not gaps:
        raise ValueError(
            f"No gap between words exceeds threshold={threshold}s -- nothing to shorten"
        )

    # Keep `target` seconds right after the word, remove the rest of the gap.
    remove_ranges = [(start + target, end) for start, end in gaps]

    result = remove_time_ranges(
        input_path, remove_ranges, output_path=output_path,
        reencode=reencode, verbose=verbose, transcript=transcript,
    )
    result.artifacts["gaps_shortened"] = str(len(gaps))
    result.artifacts["threshold"] = str(threshold)
    result.artifacts["target"] = str(target)
    return result
