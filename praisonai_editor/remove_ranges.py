"""Remove explicit time ranges from audio or video (ffmpeg splice)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from .models import EditPlan, EditResult, Segment, TranscriptResult, Word
from .plan import _create_keep_segments, _merge_overlapping
from .probe import probe_media
from .render import FFmpegAudioRenderer, FFmpegVideoRenderer

TimeSpec = Union[str, float, int]
RangeSpec = Union[str, Tuple[TimeSpec, TimeSpec]]


def parse_time(value: TimeSpec) -> float:
    """Parse a timestamp to seconds.

    Supports:
    - seconds: ``713``, ``713.5``
    - ``mm:ss``: ``11:53``
    - ``hh:mm:ss``: ``1:11:53``
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Empty time value")

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Invalid time format: {value!r}")


def parse_time_range(spec: RangeSpec) -> Tuple[float, float]:
    """Parse ``START-END`` or ``(start, end)`` into seconds."""
    if isinstance(spec, tuple):
        start, end = spec
        return parse_time(start), parse_time(end)

    text = str(spec).strip()
    for sep in ("-", "–", "—", ",", " to "):
        if sep in text:
            left, right = text.split(sep, 1)
            start, end = parse_time(left.strip()), parse_time(right.strip())
            break
    else:
        raise ValueError(
            f"Invalid range {spec!r} — use START-END (e.g. 11:53-12:43)"
        )

    if end <= start:
        raise ValueError(f"Range end must be after start: {spec!r}")
    return start, end


def build_remove_plan(
    duration: float,
    remove_ranges: Sequence[Tuple[float, float]],
) -> EditPlan:
    """Build an edit plan that removes the given time ranges."""
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if not remove_ranges:
        raise ValueError("At least one remove range is required")

    remove_segments: List[Segment] = []
    for start, end in remove_ranges:
        if start < 0:
            raise ValueError(f"Range start must be >= 0: {start}")
        if end > duration + 0.01:
            raise ValueError(
                f"Range end {end:.2f}s exceeds media duration {duration:.2f}s"
            )
        clip_start = max(0.0, start)
        clip_end = min(duration, end)
        if clip_end <= clip_start:
            continue
        remove_segments.append(
            Segment(
                start=clip_start,
                end=clip_end,
                action="remove",
                reason=f"Manual cut {clip_start:.2f}s–{clip_end:.2f}s",
                category="manual",
                confidence=1.0,
            )
        )

    if not remove_segments:
        raise ValueError("No valid remove ranges after clipping to duration")

    merged = _merge_overlapping(remove_segments)
    segments = _create_keep_segments(merged, duration)
    removed = sum(s.end - s.start for s in merged)
    kept = duration - removed

    return EditPlan(
        segments=segments,
        original_duration=duration,
        edited_duration=kept,
        removed_duration=removed,
        removal_summary={"manual": removed},
    )


def _retime_transcript(transcript: TranscriptResult, plan: EditPlan) -> TranscriptResult:
    """Re-time a transcript's words to match a remove-ranges edit plan's
    compacted output.

    Walks ``transcript.words`` against ``plan.segments`` (already merged/
    deduplicated by :func:`build_remove_plan`) rather than re-deriving a
    removed-ranges merge from scratch, so this always agrees with what the
    renderer actually kept.

    A word survives only if it falls ENTIRELY within a single kept segment
    (``word.start >= segment.start and word.end <= segment.end``); anything
    straddling a cut boundary, or entirely inside a removed gap, is dropped
    (no attempt is made to split a straddling word).

    A surviving word's new start/end is re-expressed against the compacted
    output timeline: the summed duration of every EARLIER kept segment, plus
    the word's own offset into its own segment. A word between two removed
    ranges therefore shifts by only the duration removed BEFORE it, not the
    total removed duration.
    """
    keep_segments = plan.get_keep_segments()
    retimed_words: List[Word] = []
    cumulative = 0.0
    for seg in keep_segments:
        for word in transcript.words:
            if word.start >= seg.start and word.end <= seg.end:
                offset = word.start - seg.start
                duration = word.end - word.start
                new_start = cumulative + offset
                retimed_words.append(
                    Word(
                        text=word.text,
                        start=new_start,
                        end=new_start + duration,
                        confidence=word.confidence,
                    )
                )
        cumulative += seg.end - seg.start

    return TranscriptResult(
        text=" ".join(w.text for w in retimed_words),
        words=retimed_words,
        language=transcript.language,
        duration=plan.edited_duration,
    )


def remove_time_ranges(
    input_path: str,
    remove_ranges: Sequence[RangeSpec],
    output_path: str | None = None,
    *,
    reencode: bool = False,
    verbose: bool = False,
    transcript: TranscriptResult | None = None,
    refine_boundaries: bool = True,
) -> EditResult:
    """Remove one or more time ranges from a media file.

    Args:
        input_path: Source audio or video file.
        remove_ranges: Each item is ``\"11:53-12:43\"`` or ``(\"11:53\", \"12:43\")``.
        output_path: Destination path (default: ``{stem}_cut{ext}``).
        reencode: Re-encode instead of stream copy (slower, cleaner cuts).
        verbose: Print ffmpeg progress.
        transcript: An optional transcript that was synced to ``input_path``.
            When given, the returned :class:`EditResult`'s ``transcript`` is
            this same transcript RE-TIMED to match the cut output (words
            inside a removed range dropped, later words shifted) rather than
            merely passed through unchanged. When omitted, the returned
            ``transcript`` is ``None`` (matches every existing caller).
        refine_boundaries: Only takes effect when `transcript` is given --
            nudges each range's edges to the nearest real acoustic gap
            (energy-minimum + zero-crossing search, hard-clamped to never
            cross into a neighboring word's own timestamp) before cutting.
            See boundary_refine.py's module docstring for why: an ASR word
            timestamp is a coarse hint, not a precise acoustic boundary,
            especially for fast/connected speech where words run together
            with no clean gap -- cutting at the raw reported timestamp can
            clip a neighbor's onset or leave a fragment behind, sounding
            "abrupt" even though the join itself has no click. Set False to
            cut at the raw reported timestamps unchanged (the behavior
            before this existed). A failure in the boundary search itself
            (e.g. an unreadable file) falls back to the unrefined
            timestamps rather than failing the whole edit over what is a
            quality improvement, not a correctness requirement.

    Returns:
        :class:`EditResult` with output path and edit plan.
    """
    parsed = [parse_time_range(r) for r in remove_ranges]

    if transcript is not None and refine_boundaries and transcript.words:
        from .boundary_refine import refine_range_boundaries
        try:
            parsed = refine_range_boundaries(input_path, parsed, transcript.words)
        except Exception as e:
            if verbose:
                print(f"[Warning] Boundary refinement failed, using raw timestamps: {e}")

    probe = probe_media(input_path)
    plan = build_remove_plan(probe.duration, parsed)

    src = Path(input_path)
    if output_path:
        out = Path(output_path)
    else:
        out = src.parent / f"{src.stem}_cut{src.suffix}"

    renderer = FFmpegVideoRenderer() if probe.has_video else FFmpegAudioRenderer()
    rendered = renderer.render(
        input_path,
        str(out),
        plan,
        copy_codec=not reencode,
        verbose=verbose,
    )

    retimed_transcript = _retime_transcript(transcript, plan) if transcript is not None else None

    return EditResult(
        input_path=input_path,
        output_path=rendered,
        probe=probe,
        transcript=retimed_transcript,
        plan=plan,
        success=True,
    )
