"""Edit plan creation from transcripts — heuristic and LLM-based."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .models import EditPlan, Segment, TranscriptResult, Word

# Common filler words
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean", "basically",
    "actually", "literally", "so", "well", "right", "okay", "ok",
}


def detect_fillers(words: List[Word]) -> List[Segment]:
    """Detect filler words in transcript."""
    segments = []
    for word in words:
        text_clean = re.sub(r"[^\w\s]", "", word.text.lower().strip())
        if text_clean in FILLER_WORDS:
            segments.append(Segment(
                start=word.start,
                end=word.end,
                action="remove",
                reason=f"Filler word: '{word.text}'",
                category="filler",
                text=word.text,
                confidence=0.9,
            ))
    return segments


def detect_repetitions(words: List[Word], window: int = 3) -> List[Segment]:
    """Detect repeated words/phrases."""
    segments = []
    if len(words) < 2:
        return segments

    i = 0
    while i < len(words) - 1:
        curr_clean = re.sub(r"[^\w]", "", words[i].text.lower().strip())
        if not curr_clean:
            i += 1
            continue

        j = i + 1
        while j < len(words) and j < i + window:
            next_clean = re.sub(r"[^\w]", "", words[j].text.lower().strip())
            if curr_clean == next_clean and len(curr_clean) > 2:
                segments.append(Segment(
                    start=words[i].start,
                    end=words[i].end,
                    action="remove",
                    reason=f"Repeated word: '{words[i].text}'",
                    category="repetition",
                    text=words[i].text,
                    confidence=0.85,
                ))
                break
            j += 1
        i += 1
    return segments


def detect_silences(
    words: List[Word],
    duration: float,
    min_silence: float = 1.5,
) -> List[Segment]:
    """Detect long silences between words."""
    segments = []
    if not words:
        return segments

    # Silence at start
    if words[0].start > min_silence:
        segments.append(Segment(
            start=0,
            end=words[0].start - 0.2,
            action="remove",
            reason=f"Long silence at start: {words[0].start:.1f}s",
            category="silence",
            confidence=0.95,
        ))

    # Silences between words
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap > min_silence:
            segments.append(Segment(
                start=words[i].end + 0.1,
                end=words[i + 1].start - 0.1,
                action="remove",
                reason=f"Long silence: {gap:.1f}s",
                category="silence",
                confidence=0.95,
            ))

    # Silence at end
    if words and duration - words[-1].end > min_silence:
        segments.append(Segment(
            start=words[-1].end + 0.2,
            end=duration,
            action="remove",
            reason=f"Long silence at end: {duration - words[-1].end:.1f}s",
            category="silence",
            confidence=0.95,
        ))

    return segments


def _merge_overlapping(segments: List[Segment]) -> List[Segment]:
    """Merge overlapping remove segments."""
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start <= last.end + 0.1:
            merged[-1] = Segment(
                start=last.start,
                end=max(last.end, seg.end),
                action="remove",
                reason=f"{last.reason}; {seg.reason}",
                category=last.category if last.category == seg.category else "mixed",
                confidence=min(last.confidence, seg.confidence),
            )
        else:
            merged.append(seg)
    return merged


def _create_keep_segments(
    remove_segments: List[Segment],
    duration: float,
) -> List[Segment]:
    """Create keep segments from gaps between remove segments."""
    all_segments = []
    current_time = 0.0

    for remove_seg in remove_segments:
        if remove_seg.start > current_time + 0.05:
            all_segments.append(Segment(
                start=current_time,
                end=remove_seg.start,
                action="keep",
                reason="Content",
                category="content",
                confidence=1.0,
            ))
        all_segments.append(remove_seg)
        current_time = remove_seg.end

    if current_time < duration - 0.05:
        all_segments.append(Segment(
            start=current_time,
            end=duration,
            action="keep",
            reason="Content",
            category="content",
            confidence=1.0,
        ))

    return all_segments


class HeuristicEditor:
    """Creates edit plans using heuristic detection. Implements the Editor protocol."""

    def create_plan(
        self,
        transcript: TranscriptResult,
        duration: float,
        *,
        remove_fillers: bool = True,
        remove_repetitions: bool = True,
        remove_silence: bool = True,
        min_silence: float = 1.5,
    ) -> EditPlan:
        """Create edit plan from transcript analysis."""
        remove_segments: List[Segment] = []

        if remove_fillers:
            remove_segments.extend(detect_fillers(transcript.words))
        if remove_repetitions:
            remove_segments.extend(detect_repetitions(transcript.words))
        if remove_silence:
            remove_segments.extend(detect_silences(transcript.words, duration, min_silence))

        remove_segments.sort(key=lambda s: s.start)
        merged_removes = _merge_overlapping(remove_segments)
        all_segments = _create_keep_segments(merged_removes, duration)

        removed_duration = sum(s.end - s.start for s in merged_removes)
        edited_duration = duration - removed_duration

        removal_summary: Dict[str, float] = {}
        for seg in merged_removes:
            removal_summary[seg.category] = removal_summary.get(seg.category, 0) + (seg.end - seg.start)

        return EditPlan(
            segments=all_segments,
            original_duration=duration,
            edited_duration=edited_duration,
            removed_duration=removed_duration,
            removal_summary=removal_summary,
        )


# Preset configurations
PRESETS = {
    "podcast": {
        "remove_fillers": True,
        "remove_repetitions": True,
        "remove_silence": True,
        "min_silence": 1.5,
    },
    "meeting": {
        "remove_fillers": True,
        "remove_repetitions": False,
        "remove_silence": True,
        "min_silence": 2.0,
    },
    "course": {
        "remove_fillers": True,
        "remove_repetitions": True,
        "remove_silence": True,
        "min_silence": 1.0,
    },
    "clean": {
        "remove_fillers": True,
        "remove_repetitions": True,
        "remove_silence": True,
        "min_silence": 0.8,
    },
}


def get_preset_config(preset: str) -> Dict:
    """Get configuration for a named preset."""
    return PRESETS.get(preset, {})


# Module-level convenience
def create_edit_plan(
    transcript: TranscriptResult,
    duration: float,
    *,
    preset: Optional[str] = None,
    remove_fillers: bool = True,
    remove_repetitions: bool = True,
    remove_silence: bool = True,
    min_silence: float = 1.5,
) -> EditPlan:
    """Create an edit plan using heuristic analysis.

    Args:
        transcript: Transcription result
        duration: Total media duration in seconds
        preset: Optional preset name (podcast, meeting, course, clean)
        remove_fillers: Remove filler words
        remove_repetitions: Remove repeated words
        remove_silence: Remove long silences
        min_silence: Min silence duration to remove

    Returns:
        EditPlan with segments to keep/remove
    """
    if preset:
        config = get_preset_config(preset)
        remove_fillers = config.get("remove_fillers", remove_fillers)
        remove_repetitions = config.get("remove_repetitions", remove_repetitions)
        remove_silence = config.get("remove_silence", remove_silence)
        min_silence = config.get("min_silence", min_silence)

    editor = HeuristicEditor()
    return editor.create_plan(
        transcript,
        duration,
        remove_fillers=remove_fillers,
        remove_repetitions=remove_repetitions,
        remove_silence=remove_silence,
        min_silence=min_silence,
    )
