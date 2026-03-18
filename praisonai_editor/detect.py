"""Audio content detection — classify segments as speech, music, or silence.

Supports multiple detection backends:
- "ina": inaSpeechSegmenter CNN (most accurate, requires `pip install praisonai-editor[detect]`)
- "librosa": spectral feature analysis (good, requires `pip install praisonai-editor[detect-lite]`)
- "ffmpeg": FFmpeg astats heuristics (basic, zero extra dependencies)
- "auto": tries ina → librosa → ffmpeg (first available)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import EditPlan, Segment, TranscriptResult, Word


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContentBlock:
    """A classified block of audio content."""
    start: float
    end: float
    content_type: str  # "speech", "music", "silence"
    mean_volume: float = 0.0  # RMS dB
    crest_factor: float = 0.0
    dynamic_range: float = 0.0
    zero_crossing_rate: float = 0.0
    confidence: float = 0.0  # 0-1, how confident the classifier is

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AudioMetrics:
    """Multi-metric audio analysis result from FFmpeg astats."""
    rms_level: float = -100.0
    peak_level: float = -100.0
    crest_factor: float = 0.0
    dynamic_range: float = 0.0
    zero_crossing_rate: float = 0.0


# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------

def _has_ina() -> bool:
    """Check if inaSpeechSegmenter is available."""
    try:
        import inaSpeechSegmenter  # noqa: F401
        return True
    except ImportError:
        return False


def _has_librosa() -> bool:
    """Check if librosa is available."""
    try:
        import librosa  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_detector(detector: str) -> str:
    """Resolve 'auto' to the best available detector.

    Priority: librosa (most reliable) → ina (optional, Keras 3 compat issues) → ffmpeg
    """
    if detector != "auto":
        return detector
    # librosa is the most reliable — core dependency, no Keras/TF issues
    if _has_librosa():
        return "librosa"
    if _has_ina():
        return "ina"
    return "ffmpeg"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def _extract_wav(media_path: str, tmp_dir: str) -> str:
    """Extract audio to WAV for analysis (required by ina/librosa)."""
    wav_path = os.path.join(tmp_dir, "audio.wav")
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", media_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return wav_path


# ---------------------------------------------------------------------------
# Backend 1: inaSpeechSegmenter (CNN, best accuracy)
# ---------------------------------------------------------------------------

def _classify_ina(
    media_path: str,
    duration: float,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio using inaSpeechSegmenter CNN.

    Returns full-timeline ContentBlocks (speech/music/silence).
    This is the most accurate backend — classifies the entire audio
    in a single pass using a trained CNN model.
    """
    # Compatibility shims for NumPy 2.x:
    # 1. inaSpeechSegmenter uses numpy.lib.pad (removed in NumPy 2.0+)
    import numpy
    if not hasattr(numpy.lib, 'pad'):
        numpy.lib.pad = numpy.pad

    # 2. pyannote.algorithms passes generators to np.vstack (needs list)
    try:
        import pyannote.algorithms.utils.viterbi as _viterbi
        import six.moves
        _orig_update_emission = _viterbi._update_emission
        def _patched_update_emission(emission, consecutive):
            return numpy.vstack([
                numpy.tile(e, (c, 1))
                for e, c in six.moves.zip(emission.T, consecutive)
            ]).T
        def _patched_update_constraint(constraint, consecutive):
            return numpy.vstack([
                numpy.tile(e, (c, 1))
                for e, c in six.moves.zip(constraint.T, consecutive)
            ]).T
        _viterbi._update_emission = _patched_update_emission
        _viterbi._update_constraint = _patched_update_constraint
    except ImportError:
        pass

    from inaSpeechSegmenter import Segmenter  # lazy import

    if verbose:
        print("    Using inaSpeechSegmenter (CNN) detector", flush=True)

    # inaSpeechSegmenter works directly on media files
    seg = Segmenter()
    segmentation = seg(media_path)  # returns [(label, start, end), ...]

    # Map inaSpeechSegmenter labels to our types
    label_map = {
        "speech": "speech",
        "music": "music",
        "noise": "silence",
        "noEnergy": "silence",
        # Gender labels from ina → treat as speech
        "male": "speech",
        "female": "speech",
    }

    blocks: List[ContentBlock] = []
    for label, start, end in segmentation:
        content_type = label_map.get(label, "silence")
        blocks.append(ContentBlock(
            start=float(start),
            end=float(end),
            content_type=content_type,
            confidence=0.95,  # CNN-based = high confidence
        ))

    if verbose:
        music_dur = sum(b.duration for b in blocks if b.content_type == "music")
        speech_dur = sum(b.duration for b in blocks if b.content_type == "speech")
        silence_dur = sum(b.duration for b in blocks if b.content_type == "silence")
        print(f"    INA result: {len(blocks)} segments "
              f"(music:{music_dur:.0f}s speech:{speech_dur:.0f}s silence:{silence_dur:.0f}s)",
              flush=True)

    return blocks


# ---------------------------------------------------------------------------
# Backend 2: librosa spectral features (lightweight, no TF)
# ---------------------------------------------------------------------------

def _classify_librosa(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio using librosa spectral features.

    Uses spectral centroid, spectral rolloff, zero crossing rate,
    and onset strength to distinguish music from speech/silence.
    Works on gaps between transcript speech blocks.
    """
    import librosa  # lazy import
    import numpy as np  # lazy import

    if verbose:
        print("    Using librosa (spectral) detector", flush=True)

    # Load audio once
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = _extract_wav(media_path, tmp_dir)
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

    total_samples = len(y)

    # Group transcript words into speech blocks
    speech_blocks = _group_speech_blocks(transcript.words, max_gap=speech_gap)

    if verbose:
        print(f"    Found {len(speech_blocks)} speech blocks", flush=True)

    all_blocks: List[ContentBlock] = []
    current_time = 0.0

    for speech in speech_blocks:
        if speech.start - current_time > min_block:
            gap_start = current_time
            gap_end = speech.start
            gap_duration = gap_end - gap_start

            # Extract gap audio samples
            start_sample = int(gap_start * sr)
            end_sample = min(int(gap_end * sr), total_samples)

            if end_sample > start_sample + sr // 10:  # at least 0.1s
                gap_audio = y[start_sample:end_sample]

                # Compute spectral features
                rms = float(np.sqrt(np.mean(gap_audio ** 2)))
                rms_db = 20 * math.log10(rms + 1e-10)

                if rms_db < silence_threshold:
                    content_type = "silence"
                    confidence = 0.9
                else:
                    # Spectral centroid — music is more stable
                    sc = librosa.feature.spectral_centroid(y=gap_audio, sr=sr)[0]
                    sc_mean = float(np.mean(sc))
                    sc_std = float(np.std(sc))
                    sc_cv = sc_std / (sc_mean + 1e-10)  # coefficient of variation

                    # Spectral rolloff
                    rolloff = librosa.feature.spectral_rolloff(y=gap_audio, sr=sr)[0]
                    rolloff_mean = float(np.mean(rolloff))

                    # Zero crossing rate
                    zcr = librosa.feature.zero_crossing_rate(gap_audio)[0]
                    zcr_mean = float(np.mean(zcr))

                    # Onset strength — music has more regular onsets
                    onset = librosa.onset.onset_strength(y=gap_audio, sr=sr)
                    onset_std = float(np.std(onset))

                    # Spectral flatness — music is less flat (more tonal)
                    flatness = librosa.feature.spectral_flatness(y=gap_audio)[0]
                    flatness_mean = float(np.mean(flatness))

                    # Score-based classification
                    music_score = 0.0

                    # Music has more stable spectral centroid (lower CV)
                    if sc_cv < 0.5:
                        music_score += 2.0
                    elif sc_cv < 1.0:
                        music_score += 1.0
                    else:
                        music_score -= 1.0

                    # Music has lower ZCR (tonal vs noisy)
                    if zcr_mean < 0.05:
                        music_score += 2.0
                    elif zcr_mean < 0.1:
                        music_score += 1.0
                    else:
                        music_score -= 1.0

                    # Music is less spectrally flat (more tonal)
                    if flatness_mean < 0.1:
                        music_score += 2.0
                    elif flatness_mean < 0.3:
                        music_score += 1.0

                    # Music has higher and more consistent onset strength
                    if onset_std > 0 and onset_std < 5.0:
                        music_score += 1.0

                    content_type = "music" if music_score >= 3.0 else "speech"
                    confidence = min(0.9, 0.5 + abs(music_score) * 0.1)

                all_blocks.append(ContentBlock(
                    start=gap_start,
                    end=gap_end,
                    content_type=content_type,
                    mean_volume=rms_db,
                    zero_crossing_rate=zcr_mean if content_type != "silence" else 0,
                    confidence=confidence,
                ))

                if verbose:
                    print(f"    Gap {gap_start:.1f}-{gap_end:.1f}s: {content_type} "
                          f"(RMS:{rms_db:.1f}dB ZCR:{zcr_mean if content_type != 'silence' else 0:.3f} "
                          f"conf:{confidence:.2f})", flush=True)
            else:
                all_blocks.append(ContentBlock(
                    start=current_time, end=speech.start,
                    content_type="silence", mean_volume=-100.0,
                ))

        elif speech.start > current_time:
            all_blocks.append(ContentBlock(
                start=current_time, end=speech.start,
                content_type="silence", mean_volume=-100.0,
            ))

        all_blocks.append(speech)
        current_time = speech.end

    # Handle trailing gap
    if current_time < duration - min_block:
        start_sample = int(current_time * sr)
        end_sample = min(int(duration * sr), total_samples)
        if end_sample > start_sample + sr // 10:
            gap_audio = y[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(gap_audio ** 2)))
            rms_db = 20 * math.log10(rms + 1e-10)

            if rms_db < silence_threshold:
                content_type = "silence"
            else:
                import numpy as np
                sc = librosa.feature.spectral_centroid(y=gap_audio, sr=sr)[0]
                zcr = librosa.feature.zero_crossing_rate(gap_audio)[0]
                flatness = librosa.feature.spectral_flatness(y=gap_audio)[0]
                music_score = 0.0
                if float(np.std(sc)) / (float(np.mean(sc)) + 1e-10) < 0.5:
                    music_score += 2.0
                if float(np.mean(zcr)) < 0.05:
                    music_score += 2.0
                if float(np.mean(flatness)) < 0.1:
                    music_score += 2.0
                content_type = "music" if music_score >= 3.0 else "speech"

            all_blocks.append(ContentBlock(
                start=current_time, end=duration,
                content_type=content_type, mean_volume=rms_db,
            ))
    elif current_time < duration:
        all_blocks.append(ContentBlock(
            start=current_time, end=duration,
            content_type="silence", mean_volume=-100.0,
        ))

    return all_blocks


# ---------------------------------------------------------------------------
# Backend 3: FFmpeg astats (zero-dep fallback)
# ---------------------------------------------------------------------------

def _analyze_audio(media_path: str, start: float, duration: float) -> AudioMetrics:
    """Analyze audio segment using FFmpeg astats for multiple metrics."""
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
    """Classify audio type using FFmpeg astats metrics."""
    if metrics.rms_level < silence_threshold:
        return "silence"

    music_score = 0.0

    if metrics.crest_factor > 0:
        if metrics.crest_factor < 10:
            music_score += 3.0
        elif metrics.crest_factor < 15:
            music_score += 1.5
        else:
            music_score -= 1.0

    if metrics.zero_crossing_rate > 0:
        if metrics.zero_crossing_rate < 0.05:
            music_score += 2.0
        elif metrics.zero_crossing_rate < 0.1:
            music_score += 0.5
        else:
            music_score -= 1.0

    if metrics.rms_level > -25:
        music_score += 1.0
    elif metrics.rms_level > -35:
        music_score += 0.5

    return "music" if music_score >= 2.0 else "speech"


def _classify_ffmpeg(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio using FFmpeg astats heuristics (zero-dep fallback)."""
    if verbose:
        print("    Using FFmpeg astats detector", flush=True)

    speech_blocks = _group_speech_blocks(transcript.words, max_gap=speech_gap)

    if verbose:
        print(f"    Found {len(speech_blocks)} speech blocks", flush=True)

    all_blocks: List[ContentBlock] = []
    current_time = 0.0

    for speech in speech_blocks:
        if speech.start - current_time > min_block:
            gap_start = current_time
            gap_end = speech.start
            gap_duration = gap_end - gap_start

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
                confidence=0.7,
            ))

            if verbose:
                print(f"    Gap {gap_start:.1f}-{gap_end:.1f}s: {content_type} "
                      f"(RMS:{metrics.rms_level:.1f}dB CF:{metrics.crest_factor:.1f} "
                      f"ZCR:{metrics.zero_crossing_rate:.3f})", flush=True)

        elif speech.start > current_time:
            all_blocks.append(ContentBlock(
                start=current_time, end=speech.start,
                content_type="silence", mean_volume=-100.0,
            ))

        all_blocks.append(speech)
        current_time = speech.end

    if current_time < duration - min_block:
        metrics = _analyze_audio(media_path, current_time, duration - current_time)
        content_type = _classify_by_metrics(metrics, silence_threshold)
        all_blocks.append(ContentBlock(
            start=current_time, end=duration,
            content_type=content_type,
            mean_volume=metrics.rms_level,
            crest_factor=metrics.crest_factor,
            dynamic_range=metrics.dynamic_range,
            zero_crossing_rate=metrics.zero_crossing_rate,
            confidence=0.7,
        ))
        if verbose:
            print(f"    Gap {current_time:.1f}-{duration:.1f}s: {content_type} "
                  f"(RMS:{metrics.rms_level:.1f}dB CF:{metrics.crest_factor:.1f} "
                  f"ZCR:{metrics.zero_crossing_rate:.3f})", flush=True)
    elif current_time < duration:
        all_blocks.append(ContentBlock(
            start=current_time, end=duration,
            content_type="silence", mean_volume=-100.0,
        ))

    return all_blocks


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _group_speech_blocks(
    words: List[Word],
    max_gap: float = 2.0,
) -> List[ContentBlock]:
    """Group consecutive words into speech blocks."""
    if not words:
        return []

    blocks = []
    block_start = words[0].start
    block_end = words[0].end

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > max_gap:
            blocks.append(ContentBlock(
                start=block_start, end=block_end,
                content_type="speech",
            ))
            block_start = words[i].start
        block_end = words[i].end

    blocks.append(ContentBlock(
        start=block_start, end=block_end,
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
    """
    if len(blocks) < 3:
        return blocks

    merged = [blocks[0]]

    i = 1
    while i < len(blocks):
        prev = merged[-1]
        curr = blocks[i]

        if (prev.content_type == "music"
            and i + 1 < len(blocks)
            and blocks[i + 1].content_type == "music"):

            gap = curr
            bridge_ok = False

            if gap.content_type == "speech" and gap.duration <= max_speech_bridge:
                bridge_ok = True
            elif gap.content_type == "silence" and gap.duration <= max_silence_bridge:
                bridge_ok = True

            if bridge_ok:
                next_music = blocks[i + 1]
                avg_vol = (prev.mean_volume + next_music.mean_volume) / 2
                merged[-1] = ContentBlock(
                    start=prev.start,
                    end=next_music.end,
                    content_type="music",
                    mean_volume=avg_vol,
                    confidence=max(prev.confidence, next_music.confidence),
                )
                if verbose:
                    print(f"    Merged music: {prev.start:.1f}-{next_music.end:.1f}s "
                          f"(bridged {gap.duration:.1f}s {gap.content_type})", flush=True)
                i += 2
                continue

        merged.append(curr)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def classify_content(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    detector: str = "auto",
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    max_speech_bridge: float = 30.0,
    max_silence_bridge: float = 5.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio content into speech, music, and silence blocks.

    Args:
        media_path: Path to media file
        transcript: Transcription result with word timestamps
        duration: Total duration in seconds
        detector: Detection backend — "auto", "ina", "librosa", or "ffmpeg"
        speech_gap: Max gap (s) between words for same speech block
        silence_threshold: RMS dB below which = silence
        min_block: Min gap duration to analyze
        max_speech_bridge: Max speech gap (s) between music blocks to merge
        max_silence_bridge: Max silence gap (s) between music blocks to merge
        verbose: Print progress

    Returns:
        List of ContentBlocks covering the full duration
    """
    resolved = _resolve_detector(detector)

    if verbose:
        print(f"    Detector: {resolved}" + (" (auto-selected)" if detector == "auto" else ""),
              flush=True)

    # Dispatch to backend
    if resolved == "ina":
        all_blocks = _classify_ina(media_path, duration, verbose=verbose)
    elif resolved == "librosa":
        all_blocks = _classify_librosa(
            media_path, transcript, duration,
            speech_gap=speech_gap,
            silence_threshold=silence_threshold,
            min_block=min_block,
            verbose=verbose,
        )
    elif resolved == "ffmpeg":
        all_blocks = _classify_ffmpeg(
            media_path, transcript, duration,
            speech_gap=speech_gap,
            silence_threshold=silence_threshold,
            min_block=min_block,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown detector: {resolved!r}. "
                         f"Use 'auto', 'ina', 'librosa', or 'ffmpeg'.")

    # Post-process: merge fragmented music blocks
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
    detector: str = "auto",
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> Tuple[EditPlan, List[ContentBlock]]:
    """Create an edit plan based on content classification.

    Returns:
        Tuple of (EditPlan, List[ContentBlock])
    """
    blocks = classify_content(
        media_path, transcript, duration,
        detector=detector,
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
            confidence=block.confidence,
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
