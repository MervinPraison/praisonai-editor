"""FFT-based adaptive noise reduction for background hiss/hum.

Uses ffmpeg's ``afftdn`` filter — an FFT adaptive denoiser built into
ffmpeg with no external model file required (unlike ``arnndn``, which
needs a separate RNN model download). This is the "Studio Sound"-style
denoise step; ``normalize``/``master`` only touch loudness, not noise.

Usage:
    from praisonai_editor.denoise import denoise_audio

    result = denoise_audio("noisy.m4a")                       # default settings
    result = denoise_audio("noisy.m4a", noise_reduction=20)   # stronger reduction
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .convert import _find_ffmpeg
from .models import EditResult

#: afftdn's own valid AVOption ranges.
_MIN_NOISE_REDUCTION = 0.01
_MAX_NOISE_REDUCTION = 97.0
_MIN_NOISE_FLOOR = -80.0
_MAX_NOISE_FLOOR = -20.0


def _build_denoise_filter(noise_reduction: float, noise_floor: float, track_noise: bool) -> str:
    """Build the -af filter chain for FFT-based noise reduction (afftdn)."""
    tn = 1 if track_noise else 0
    return f"afftdn=nr={noise_reduction:g}:nf={noise_floor:g}:tn={tn}"


def denoise_audio(
    input_path: str,
    output_path: str | None = None,
    *,
    noise_reduction: float = 12.0,
    noise_floor: float = -50.0,
    track_noise: bool = True,
    bitrate: str = "192k",
    verbose: bool = False,
) -> EditResult:
    """Reduce background noise (hiss/hum) with ffmpeg's FFT denoiser (afftdn).

    Args:
        input_path: Source audio file.
        output_path: Destination path (default: ``{stem}_denoised.m4a`` next
            to the input — output is always AAC, so, like ``conform``/
            ``master``, the default name fixes the extension rather than
            reusing the source's; an arbitrary source extension carrying
            AAC data round-trips unreliably in ffmpeg's demuxers).
        noise_reduction: Amount of reduction in dB (0.01–97, default 12).
        noise_floor: Expected noise floor in dB (-80 to -20, default -50).
        track_noise: Adapt to noise that changes over the file (default
            True — real recordings rarely have a perfectly constant noise
            floor, so this gives a better one-click result than a static
            profile).
        bitrate: AAC bitrate (e.g. "128k", "192k").
        verbose: Print ffmpeg progress.

    Returns:
        :class:`EditResult` with the denoised output path.
    """
    if not (_MIN_NOISE_REDUCTION <= noise_reduction <= _MAX_NOISE_REDUCTION):
        raise ValueError(
            f"noise_reduction must be between {_MIN_NOISE_REDUCTION} and "
            f"{_MAX_NOISE_REDUCTION}, got {noise_reduction}"
        )
    if not (_MIN_NOISE_FLOOR <= noise_floor <= _MAX_NOISE_FLOOR):
        raise ValueError(
            f"noise_floor must be between {_MIN_NOISE_FLOOR} and "
            f"{_MAX_NOISE_FLOOR}, got {noise_floor}"
        )

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out = Path(output_path) if output_path else src.parent / f"{src.stem}_denoised.m4a"
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()
    af = _build_denoise_filter(noise_reduction, noise_floor, track_noise)
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
        raise RuntimeError(f"FFmpeg denoise failed: {stderr[-800:]}")

    return EditResult(
        input_path=str(src),
        output_path=str(out),
        success=True,
        artifacts={
            "filter": af,
            "noise_reduction": str(noise_reduction),
            "noise_floor": str(noise_floor),
            "track_noise": str(track_noise),
        },
    )
