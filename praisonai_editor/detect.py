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
    content_type: str  # "speech", "music", "silence", "noise", etc.
    detector: str = "auto"  # which backend generated this block
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

                content_type, confidence, rms_db, zcr_mean = _librosa_analyze_segment(
                    gap_audio, sr, silence_threshold
                )

                all_blocks.append(ContentBlock(
                    start=gap_start,
                    end=gap_end,
                    content_type=content_type,
                    detector="librosa",
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
            content_type, confidence, rms_db, zcr_mean = _librosa_analyze_segment(
                gap_audio, sr, silence_threshold
            )
            all_blocks.append(ContentBlock(
                start=current_time,
                end=duration,
                content_type=content_type,
                detector="librosa",
                mean_volume=rms_db,
                zero_crossing_rate=zcr_mean if content_type != "silence" else 0,
                confidence=confidence,
            ))
            if verbose:
                print(f"    Gap {current_time:.1f}-{duration:.1f}s: {content_type} "
                      f"(RMS:{rms_db:.1f}dB ZCR:{zcr_mean if content_type != 'silence' else 0:.3f} "
                      f"conf:{confidence:.2f})", flush=True)
        else:
            all_blocks.append(ContentBlock(
                start=current_time, end=duration,
                content_type="silence", mean_volume=-100.0,
            ))

    return all_blocks


def _librosa_analyze_segment(
    audio_segment, sr: int, silence_threshold: float
) -> Tuple[str, float, float, float]:
    """Helper to analyze a specific audio segment using librosa."""
    import librosa
    import numpy as np
    import math

    rms = float(np.sqrt(np.mean(audio_segment ** 2)))
    rms_db = 20 * math.log10(rms + 1e-10)

    if rms_db < silence_threshold:
        return "silence", 0.9, rms_db, 0.0

    sc = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
    sc_mean = float(np.mean(sc))
    sc_std = float(np.std(sc))
    sc_cv = sc_std / (sc_mean + 1e-10)

    zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
    zcr_mean = float(np.mean(zcr))

    onset = librosa.onset.onset_strength(y=audio_segment, sr=sr)
    onset_std = float(np.std(onset))

    flatness = librosa.feature.spectral_flatness(y=audio_segment)[0]
    flatness_mean = float(np.mean(flatness))

    music_score = 0.0

    if sc_cv < 0.5: music_score += 2.0
    elif sc_cv < 1.0: music_score += 1.0
    else: music_score -= 1.0

    if zcr_mean < 0.05: music_score += 2.0
    elif zcr_mean < 0.1: music_score += 1.0
    else: music_score -= 1.0

    if flatness_mean < 0.1: music_score += 2.0
    elif flatness_mean < 0.3: music_score += 1.0

    if onset_std > 0 and onset_std < 5.0: music_score += 1.0

    content_type = "music" if music_score >= 3.0 else "speech"
    confidence = min(0.9, 0.5 + abs(music_score) * 0.1)

    return content_type, confidence, rms_db, zcr_mean


def _classify_librosa_full(
    media_path: str,
    duration: float,
    *,
    window_size: float = 2.0,
    silence_threshold: float = -45.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Classify audio using librosa over the entire timeline using a sliding window."""
    import librosa
    
    if verbose:
        print("    Using librosa (full timeline) detector", flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = _extract_wav(media_path, tmp_dir)
        y, sr = librosa.load(wav_path, sr=16000, mono=True)

    blocks: List[ContentBlock] = []
    total_samples = len(y)
    window_samples = int(window_size * sr)

    for start_sample in range(0, total_samples, window_samples):
        end_sample = min(start_sample + window_samples, total_samples)
        segment = y[start_sample:end_sample]
        start_time = start_sample / sr
        end_time = end_sample / sr

        if len(segment) < sr // 10:
            continue

        content_type, confidence, rms_db, zcr_mean = _librosa_analyze_segment(
            segment, sr, silence_threshold
        )

        blocks.append(ContentBlock(
            start=start_time,
            end=end_time,
            content_type=content_type,
            detector="librosa",
            mean_volume=rms_db,
            zero_crossing_rate=zcr_mean if content_type != "silence" else 0,
            confidence=confidence,
        ))

    return blocks


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


def _extract_all_events(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    silence_threshold: float = -45.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Run all available detectors to extract overlapping events across the timeline."""
    events: List[ContentBlock] = []

    # 1. LAYER: Whisper (Speech)
    speech_blocks = _group_speech_blocks(transcript.words, max_gap=2.0)
    for b in speech_blocks:
        b.detector = "whisper"
        events.append(b)

    # 2. LAYER: librosa (Spectral - full timeline)
    if _has_librosa():
        try:
            librosa_blocks = _classify_librosa_full(
                media_path, duration, silence_threshold=silence_threshold, verbose=verbose
            )
            events.extend(librosa_blocks)
        except Exception as e:
            if verbose:
                print(f"    [Warning] librosa full timeline failed: {e}", flush=True)
    else:
        # Fallback to ffmpeg gaps
        ffmpeg_blocks = _classify_ffmpeg(media_path, transcript, duration, verbose=verbose)
        for b in ffmpeg_blocks:
            if b.content_type != "silence":
                b.detector = "ffmpeg"
                events.append(b)

    # 3. LAYER: inaSpeechSegmenter (CNN - full timeline)
    if _has_ina():
        try:
            ina_blocks = _classify_ina(media_path, duration, verbose=verbose)
            for b in ina_blocks:
                b.detector = "ina"
                events.append(b)
        except Exception as e:
            if verbose:
                print(f"    [Warning] inaSpeechSegmenter failed: {e}", flush=True)

    return sorted(events, key=lambda x: x.start)

def _ensemble_decision(
    events: List[ContentBlock],
    duration: float,
    chunk_size: float = 1.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Resolve overlapping events into a single unified timeline of distinct blocks."""
    import numpy as np

    if verbose:
        print("    Running ensemble decision engine to resolve overlaps...", flush=True)

    resolved: List[ContentBlock] = []
    
    for t in np.arange(0, duration, chunk_size):
        chunk_start = float(t)
        chunk_end = min(t + chunk_size, duration)

        active_events = []
        for e in events:
            overlap_start = max(chunk_start, e.start)
            overlap_end = min(chunk_end, e.end)
            if overlap_end - overlap_start > (chunk_end - chunk_start) * 0.4:
                active_events.append(e)

        is_speech = any(e.content_type == "speech" and e.detector in ("whisper", "ina") for e in active_events)
        is_music = any(e.content_type == "music" and e.detector in ("ina", "librosa", "ffmpeg") for e in active_events)

        if is_speech and is_music:
            final_type = "speech_over_music"
        elif is_speech:
            final_type = "speech"
        elif is_music:
            final_type = "music"
        else:
            final_type = "silence"

        # Combine contiguous identical chunks
        if resolved and resolved[-1].content_type == final_type:
            resolved[-1].end = chunk_end
        else:
            resolved.append(ContentBlock(
                start=chunk_start,
                end=chunk_end,
                content_type=final_type,
                detector="ensemble",
                confidence=1.0,
            ))

    return resolved
def _merge_music_blocks(
    blocks: List[ContentBlock],
    max_speech_bridge: float = 30.0,
    max_silence_bridge: float = 5.0,
    verbose: bool = False,
) -> List[ContentBlock]:
    """Merge adjacent music blocks separated by short gaps."""
    if len(blocks) < 3:
        return blocks

    merged = [blocks[0]]

    i = 1
    while i < len(blocks):
        prev = merged[-1]
        curr = blocks[i]

        # Consider both pure music and speech_over_music as "music" for bridging
        is_prev_music = "music" in prev.content_type
        is_next_music = (i + 1 < len(blocks) and "music" in blocks[i + 1].content_type)

        if is_prev_music and is_next_music:
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



def classify_content(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    detector: str = "ensemble",
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> Tuple[List[ContentBlock], List[ContentBlock]]:
    """Classify audio content, returning (resolved_blocks, all_raw_events)."""
    backend = _resolve_detector(detector)

    if backend == "ensemble" or detector == "ensemble" or detector == "auto":
        all_events = _extract_all_events(media_path, transcript, duration, silence_threshold, verbose)
        resolved = _ensemble_decision(all_events, duration, verbose=verbose)
        resolved = _merge_music_blocks(resolved, verbose=verbose)
        return resolved, all_events

    # Backwards compatibility for single detectors
    all_events = []
    if backend == "ina":
        blocks = _classify_ina(media_path, duration, verbose=verbose)
        for b in blocks: b.detector = "ina"
        all_events = blocks
    elif backend == "librosa":
        blocks = _classify_librosa(
            media_path, transcript, duration,
            speech_gap=speech_gap, silence_threshold=silence_threshold,
            min_block=min_block, verbose=verbose
        )
        for b in blocks: b.detector = "librosa"
        all_events = blocks
    else:
        blocks = _classify_ffmpeg(
            media_path, transcript, duration,
            speech_gap=speech_gap, silence_threshold=silence_threshold,
            min_block=min_block, verbose=verbose
        )
        for b in blocks: b.detector = "ffmpeg"
        all_events = blocks

    if backend != "ina":
        blocks = _merge_music_blocks(blocks, verbose=verbose)

    return blocks, all_events


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------



def create_content_plan(
    media_path: str,
    transcript: TranscriptResult,
    duration: float,
    *,
    keep_types: List[str] = ["music"],
    detector: str = "auto",
    speech_gap: float = 2.0,
    silence_threshold: float = -45.0,
    min_block: float = 1.0,
    verbose: bool = False,
) -> Tuple[EditPlan, List[ContentBlock], List[ContentBlock]]:
    """Create an edit plan based on content classification.

    Returns:
        Tuple of (EditPlan, List[ContentBlock], List[ContentBlock])
    """
    blocks, all_events = classify_content(
        media_path, transcript, duration,
        detector=detector,
        speech_gap=speech_gap,
        silence_threshold=silence_threshold,
        min_block=min_block,
        verbose=verbose,
    )

    segments = []
    for block in blocks:
        # Check if ANY of the keeping types match the block's content_type
        # e.g. keep_types=["music"], block="speech_over_music" -> keeps it
        action = "remove"
        for t in keep_types:
            if t in block.content_type:
                action = "keep"
                break
            
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

    return plan, blocks, all_events
