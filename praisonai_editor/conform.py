"""Conform externally-mastered audio for safe splicing (rate/layout/length).

Usage:
    from praisonai_editor.conform import conform_audio

    # Resample to 48 kHz stereo AAC, default {stem}_conformed.m4a
    conform_audio("mastered.wav")

    # Force an exact length (trim if longer, pad silence if shorter)
    conform_audio("jingle.mp3", "jingle_fit.m4a", duration=12.5)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "ffmpeg not found. Install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
    )


def _build_conform_filter(
    sample_rate: int,
    channels: int,
    duration: float | None,
) -> str:
    """Build the -af filter chain for conforming audio."""
    layout = "stereo" if channels == 2 else "mono"
    af = f"aformat=sample_rates={sample_rate}:channel_layouts={layout}"
    if duration is not None:
        af += f",atrim=0:{duration},apad=whole_dur={duration}"
    return af


def conform_audio(
    input_path: str,
    output_path: str | None = None,
    *,
    sample_rate: int = 48000,
    channels: int = 2,
    bitrate: str = "192k",
    duration: float | None = None,
    verbose: bool = False,
) -> str:
    """Make an externally-mastered file safe for splicing.

    Resamples to ``sample_rate``, sets the channel layout, and — when
    ``duration`` is given — forces an EXACT length (trims if longer, pads
    silence if shorter). Output is AAC.

    Args:
        input_path: Source audio file.
        output_path: Destination path (default: ``{stem}_conformed.m4a`` next
            to the input).
        sample_rate: Target sample rate in Hz.
        channels: Target channel count — 1 (mono) or 2 (stereo).
        bitrate: AAC bitrate (e.g. "128k", "192k").
        duration: Exact output length in seconds (must be > 0 when given).
        verbose: Print ffmpeg progress.

    Returns:
        Path to the conformed file.
    """
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    if duration is not None and duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path:
        out = Path(output_path)
    else:
        out = src.parent / f"{src.stem}_conformed.m4a"
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()
    af = _build_conform_filter(sample_rate, channels, duration)
    cmd = [
        ffmpeg, "-y", "-nostdin",
        "-i", str(src),
        "-af", af,
        "-c:a", "aac",
        "-b:a", bitrate,
        str(out),
    ]

    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode() if getattr(result, "stderr", None) else ""
        raise RuntimeError(f"FFmpeg conform failed: {stderr[-800:]}")

    return str(out)
