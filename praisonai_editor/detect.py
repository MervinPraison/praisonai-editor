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
    mean_volume: float = 0.0  # RMS dB
    crest_factor: float = 0.0  # Peak/RMS ratio — lower = more compressed (music)
    dynamic_range: float = 0.0  # dB range — music has moderate, speech has high
    zero_crossing_rate: float = 0.0  # Higher = more noise/speech fricatives

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


@dataclass
class AudioMetrics:
    """Multi-metric audio analysis result from FFmpeg astats."""
    rms_level: float = -100.0  # RMS level in dB (avg loudness)
    peak_level: float = -100.0  # Peak level in dB
    crest_factor: float = 0.0  # Peak/RMS — lower = more compressed (music-like)
    dynamic_range: float = 0.0  # dB — difference between loud and quiet
    zero_crossing_rate: float = 0.0  # Crossings/sec — higher = noisier/speech


def _analyze_audio(media_path: str, start: float, duration: float) -> AudioMetrics:
    """Analyze audio segment using FFmpeg astats for multiple metrics.

    Returns AudioMetrics with:
    - rms_level: Average loudness (silence < -50dB, music/speech -30 to -10dB)
    - crest_factor: Peak/RMS ratio. Music is more compressed (lower crest ~5-15),
                    speech is more dynamic (higher crest ~10-25)
    - dynamic_range: dB range. Silence ~0, speech high, music moderate
    - zero_crossing_rate: Higher for noise/speech fricatives, lower for tonal music
    """
    if duration < 0.1:
        return AudioMetrics()

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-ss", str(start),
        "-t", str(duration),
        "-i", media_path,
        "-af", "astats",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    metrics = AudioMetrics()

    # Parse astats text output: [Parsed_astats_0 @ addr] Key: value
    # We want the LAST occurrence of each key (Overall channel stats)
    for line in stderr.split("\n"):
        if "astats" not in line:
            continue
        if "RMS level dB:" in line:
            try:
                val = line.split("RMS level dB:")[-1].strip()
                if val != "-inf":
                    metrics.rms_level = float(val)
            except ValueError:
                pass
        elif "Peak level dB:" in line:
            try:
                val = line.split("Peak level dB:")[-1].strip()
                if val != "-inf":
                    metrics.peak_level = float(val)
            except ValueError:
                pass
        elif "Crest factor:" in line:
            try:
                val = line.split("Crest factor:")[-1].strip()
                if val != "-inf" and val != "inf":
                    metrics.crest_factor = float(val)
            except ValueError:
                pass
        elif "Dynamic range:" in line:
            try:
                val = line.split("Dynamic range:")[-1].strip()
                if val != "-inf" and val != "inf":
                    metrics.dynamic_range = float(val)
            except ValueError:
                pass
        elif "Zero crossings rate:" in line:
            try:
                val = line.split("Zero crossings rate:")[-1].strip()
                metrics.zero_crossing_rate = float(val)
            except ValueError:
                pass

    return metrics


def _classify_by_metrics(metrics: AudioMetrics, silence_threshold: float = -45.0) -> str:
    """Classify audio type using multiple metrics.

    Decision logic:
    - Silence: very low RMS level
    - Music: moderate-high RMS, lower crest factor (compressed), lower ZCR
    - Speech: moderate RMS, higher crest factor (dynamic), higher ZCR
    """
    # Clear silence
    if metrics.rms_level < silence_threshold:
        return "silence"

    # Score-based classification
    music_score = 0.0

    # Music tends to have lower crest factor (more compressed/consistent energy)
    if metrics.crest_factor > 0:
        if metrics.crest_factor < 10:
            music_score += 3.0  # Very compressed — likely music
        elif metrics.crest_factor < 15:
            music_score += 1.5  # Moderately compressed
        else:
            music_score -= 1.0  # Very dynamic — likely speech

    # Music tends to have lower zero crossing rate (tonal content)
    if metrics.zero_crossing_rate > 0:
        if metrics.zero_crossing_rate < 0.05:
            music_score += 2.0  # Low ZCR — tonal/musical
        elif metrics.zero_crossing_rate < 0.1:
            music_score += 0.5  # Moderate
        else:
            music_score -= 1.0  # High ZCR — speech/noise

    # Higher RMS = more likely content (both music and speech)
    if metrics.rms_level > -25:
        music_score += 1.0  # Loud content
    elif metrics.rms_level > -35:
        music_score += 0.5

    return "music" if music_score >= 2.0 else "speech"


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


def _merge_music_blocks(
    blocks: List[ContentBlock],
    max_speech_bridge: float = 30.0,
    max_silence_bridge: float = 5.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Merge adjacent music blocks separated by short speech or silence gaps.

    Songs run for minutes — Whisper may detect a few words in lyrics,
    creating tiny speech blocks that fragment the music. This step
    consolidates them.

    Rules:
    - If two music blocks are separated by speech < max_speech_bridge seconds,
      absorb the speech into the music (Whisper detected lyrics).
    - If two music blocks are separated by silence < max_silence_bridge seconds,
      absorb the silence (instrumental pause).

    Args:
        blocks: Raw classified blocks
        max_speech_bridge: Max speech duration (s) between music blocks to merge (default: 30s)
        max_silence_bridge: Max silence duration (s) between music blocks to merge (default: 5s)
        verbose: Print merge info
    """
    if len(blocks) < 3:
        return blocks

    merged = [blocks[0]]

    i = 1
    while i < len(blocks):
        prev = merged[-1]
        curr = blocks[i]

        # Check if we can merge: music - [short gap] - music
        if (prev.content_type == "music"
            and i + 1 < len(blocks)
            and blocks[i + 1].content_type == "music"):

            gap = curr  # The block between two music blocks
            bridge_ok = False

            if gap.content_type == "speech" and gap.duration <= max_speech_bridge:
                bridge_ok = True  # Whisper probably picked up lyrics
            elif gap.content_type == "silence" and gap.duration <= max_silence_bridge:
                bridge_ok = True  # Short instrumental pause

            if bridge_ok:
                # Merge: extend prev music to cover gap + next music
                next_music = blocks[i + 1]
                avg_vol = (prev.mean_volume + next_music.mean_volume) / 2
                merged[-1] = ContentBlock(
                    start=prev.start,
                    end=next_music.end,
                    content_type="music",
                    mean_volume=avg_vol,
                )
                if verbose:
                    print(f"    Merged music: {prev.start:.1f}-{next_music.end:.1f}s "
                          f"(bridged {gap.duration:.1f}s {gap.content_type})", flush=True)
                i += 2  # Skip gap + next music (absorbed)
                continue

        merged.append(curr)
        i += 1

    return merged


def classify_content(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    max_speech_bridge: float = 30.0,
    max_silence_bridge: float = 5.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio content into speech, music, and silence blocks.

    Algorithm:
    1. Group transcript words into speech blocks (gaps > speech_gap = new block)
    2. For each non-speech gap, measure audio energy via FFmpeg
    3. High energy gaps = music, low energy gaps = silence
    4. Merge adjacent music blocks separated by short speech/silence (songs run for minutes)

    Args:
        media_path: Path to media file
        transcript: Transcription result with word timestamps
        duration: Total duration in seconds
        speech_gap: Max gap (s) between words to be same speech block
        silence_threshold: Volume (dB) below which = silence (default: -45dB)
        min_block: Minimum block duration to analyze
        max_speech_bridge: Max speech gap (s) between music blocks to merge (lyrics)
        max_silence_bridge: Max silence gap (s) between music blocks to merge
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

            # Analyze audio in this gap with multiple metrics
            metrics = _analyze_audio(media_path, gap_start, gap_duration)
            content_type = _classify_by_metrics(metrics, silence_threshold)

            all_blocks.append(ContentBlock(
                start=gap_start,
                end=gap_end,
                content_type=content_type,
                mean_volume=metrics.rms_level,
                crest_factor=metrics.crest_factor,
                dynamic_range=metrics.dynamic_range,
                zero_crossing_rate=metrics.zero_crossing_rate,
            ))

            if verbose:
                print(f"    Gap {gap_start:.1f}-{gap_end:.1f}s: {content_type} "
                      f"(RMS:{metrics.rms_level:.1f}dB CF:{metrics.crest_factor:.1f} "
                      f"ZCR:{metrics.zero_crossing_rate:.3f})", flush=True)

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
        metrics = _analyze_audio(media_path, current_time, duration - current_time)
        content_type = _classify_by_metrics(metrics, silence_threshold)
        all_blocks.append(ContentBlock(
            start=current_time,
            end=duration,
            content_type=content_type,
            mean_volume=metrics.rms_level,
            crest_factor=metrics.crest_factor,
            dynamic_range=metrics.dynamic_range,
            zero_crossing_rate=metrics.zero_crossing_rate,
        ))
        if verbose:
            print(f"    Gap {current_time:.1f}-{duration:.1f}s: {content_type} "
                  f"(RMS:{metrics.rms_level:.1f}dB CF:{metrics.crest_factor:.1f} "
                  f"ZCR:{metrics.zero_crossing_rate:.3f})", flush=True)
    elif current_time < duration:
        all_blocks.append(ContentBlock(
            start=current_time,
            end=duration,
            content_type="silence",
            mean_volume=-100.0,
        ))

    # Step 3: Merge fragmented music blocks (songs run for minutes, not seconds)
    raw_count = len(all_blocks)
    all_blocks = _merge_music_blocks(
        all_blocks,
        max_speech_bridge=max_speech_bridge,
        max_silence_bridge=max_silence_bridge,
        verbose=verbose,
    )
    if verbose and len(all_blocks) < raw_count:
        print(f"    Merged: {raw_count} → {len(all_blocks)} blocks", flush=True)

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
) -> Tuple[EditPlan, List[ContentBlock]]:
    """Create an edit plan based on content classification.

    Returns:
        Tuple of (EditPlan, List[ContentBlock]) — the plan and the raw detection blocks
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

    plan = EditPlan(
        segments=segments,
        original_duration=duration,
        edited_duration=kept,
        removed_duration=removed,
        removal_summary=removal_summary,
    )

    return plan, blocks
