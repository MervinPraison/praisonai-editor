"""Remove explicit time ranges from audio or video (ffmpeg splice)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from .models import EditPlan, EditResult, Segment
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


def remove_time_ranges(
    input_path: str,
    remove_ranges: Sequence[RangeSpec],
    output_path: str | None = None,
    *,
    reencode: bool = False,
    verbose: bool = False,
) -> EditResult:
    """Remove one or more time ranges from a media file.

    Args:
        input_path: Source audio or video file.
        remove_ranges: Each item is ``\"11:53-12:43\"`` or ``(\"11:53\", \"12:43\")``.
        output_path: Destination path (default: ``{stem}_cut{ext}``).
        reencode: Re-encode instead of stream copy (slower, cleaner cuts).
        verbose: Print ffmpeg progress.

    Returns:
        :class:`EditResult` with output path and edit plan.
    """
    parsed = [parse_time_range(r) for r in remove_ranges]
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

    return EditResult(
        input_path=input_path,
        output_path=rendered,
        probe=probe,
        plan=plan,
        success=True,
    )
