"""Audio content detection — classify segments as speech, music, or silence.

Uses transcript word timestamps + FFmpeg energy analysis to distinguish
music from silence in non-speech gaps.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import EditPlan, Segment, TranscriptResult, Word


@dataclass
class ContentBlock:
    """A classified block of audio content."""
    start: float
    end: float
    content_type: str  # "speech", "music", "silence"
    mean_volume: float = 0.0  # dB

    @property
    def duration(self) -> float:
        return self.end - self.start


def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def _measure_volume(media_path: str, start: float, duration: float) -> float:
    """Measure mean volume (dB) for a segment of audio using FFmpeg.

    Returns mean_volume in dB. Typical values:
    - Silence: < -50 dB
    - Music: -30 to -10 dB
    - Speech: -25 to -10 dB
    """
    if duration < 0.1:
        return -100.0

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-ss", str(start),
        "-t", str(duration),
        "-i", media_path,
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse mean_volume from FFmpeg output
    for line in stderr.split("\n"):
        if "mean_volume:" in line:
            try:
                vol = float(line.split("mean_volume:")[1].strip().split()[0])
                return vol
            except (ValueError, IndexError):
                pass

    return -100.0


def _group_speech_blocks(
    words: List[Word],
    max_gap: float = 2.0,
) -> List[ContentBlock]:
    """Group consecutive words into speech blocks.

    Words within max_gap seconds of each other are considered same speech block.
    """
    if not words:
        return []

    blocks = []
    block_start = words[0].start
    block_end = words[0].end

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > max_gap:
            # End current block, start new one
            blocks.append(ContentBlock(
                start=block_start,
                end=block_end,
                content_type="speech",
            ))
            block_start = words[i].start
        block_end = words[i].end

    # Last block
    blocks.append(ContentBlock(
        start=block_start,
        end=block_end,
        content_type="speech",
    ))

    return blocks


def classify_content(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio content into speech, music, and silence blocks.

    Algorithm:
    1. Group transcript words into speech blocks (gaps > speech_gap = new block)
    2. For each non-speech gap, measure audio energy via FFmpeg
    3. High energy gaps = music, low energy gaps = silence

    Args:
        media_path: Path to media file
        transcript: Transcription result with word timestamps
        duration: Total duration in seconds
        speech_gap: Max gap (s) between words to be same speech block
        silence_threshold: Volume (dB) below which = silence (default: -45dB)
        min_block: Minimum block duration to analyze
        verbose: Print progress

    Returns:
        List of ContentBlocks covering the full duration
    """
    # Step 1: Group words into speech blocks
    speech_blocks = _group_speech_blocks(transcript.words, max_gap=speech_gap)

    if verbose:
        print(f"    Found {len(speech_blocks)} speech blocks", flush=True)

    # Step 2: Identify all gaps (before first, between, after last)
    all_blocks: List[ContentBlock] = []
    current_time = 0.0

    for speech in speech_blocks:
        # Gap before this speech block
        if speech.start - current_time > min_block:
            gap_start = current_time
            gap_end = speech.start
            gap_duration = gap_end - gap_start

            # Measure volume in this gap
            vol = _measure_volume(media_path, gap_start, gap_duration)

            content_type = "music" if vol > silence_threshold else "silence"
            all_blocks.append(ContentBlock(
                start=gap_start,
                end=gap_end,
                content_type=content_type,
                mean_volume=vol,
            ))

            if verbose:
                print(f"    Gap {gap_start:.1f}-{gap_end:.1f}s: {content_type} ({vol:.1f}dB)", flush=True)

        elif speech.start > current_time:
            # Small gap — classify as transition
            all_blocks.append(ContentBlock(
                start=current_time,
                end=speech.start,
                content_type="silence",
                mean_volume=-100.0,
            ))

        # Add speech block
        all_blocks.append(speech)
        current_time = speech.end

    # Handle gap after last speech block
    if current_time < duration - min_block:
        vol = _measure_volume(media_path, current_time, duration - current_time)
        content_type = "music" if vol > silence_threshold else "silence"
        all_blocks.append(ContentBlock(
            start=current_time,
            end=duration,
            content_type=content_type,
            mean_volume=vol,
        ))
        if verbose:
            print(f"    Gap {current_time:.1f}-{duration:.1f}s: {content_type} ({vol:.1f}dB)", flush=True)
    elif current_time < duration:
        all_blocks.append(ContentBlock(
            start=current_time,
            end=duration,
            content_type="silence",
            mean_volume=-100.0,
        ))

    return all_blocks


def create_content_plan(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    keep_types: List[str],
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> EditPlan:
    """Create an edit plan based on content classification.

    Args:
        media_path: Path to media
        transcript: Transcription result
        duration: Total duration
        keep_types: List of content types to keep ("speech", "music", "silence")
        speech_gap: Gap threshold for speech block grouping
        silence_threshold: Volume threshold for music vs silence
        min_block: Min gap duration to analyze
        verbose: Print progress

    Returns:
        EditPlan with keep/remove segments
    """
    blocks = classify_content(
        media_path, transcript, duration,
        speech_gap=speech_gap,
        silence_threshold=silence_threshold,
        min_block=min_block,
        verbose=verbose,
    )

    segments = []
    for block in blocks:
        action = "keep" if block.content_type in keep_types else "remove"
        segments.append(Segment(
            start=block.start,
            end=block.end,
            action=action,
            reason=f"{block.content_type} ({block.mean_volume:.1f}dB)",
            category=block.content_type,
            confidence=0.9,
        ))

    kept = sum(s.end - s.start for s in segments if s.action == "keep")
    removed = sum(s.end - s.start for s in segments if s.action == "remove")

    removal_summary: Dict[str, float] = {}
    for s in segments:
        if s.action == "remove":
            removal_summary[s.category] = removal_summary.get(s.category, 0) + (s.end - s.start)

    return EditPlan(
        segments=segments,
        original_duration=duration,
        edited_duration=kept,
        removed_duration=removed,
        removal_summary=removal_summary,
    )
