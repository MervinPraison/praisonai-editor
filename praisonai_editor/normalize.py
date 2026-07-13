"""Conditional loudness normalisation for quiet sermon / YouTube audio."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .convert import _find_ffmpeg


@dataclass(frozen=True)
class VolumeStats:
    mean_db: float
    max_db: float


@dataclass(frozen=True)
class NormalizeResult:
    path: str
    mean_db: float
    max_db: float
    normalized: bool
    target_lufs: float = -16.0
    true_peak_db: float = -1.5


def _parse_volumedetect(stderr: str) -> VolumeStats:
    mean_m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
    max_m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", stderr)
    if not mean_m or not max_m:
        raise RuntimeError(f"volumedetect parse failed: {stderr[-800:]}")
    return VolumeStats(float(mean_m.group(1)), float(max_m.group(1)))


def measure_volume(path: str) -> VolumeStats:
    """Return mean and peak volume (dB) via ffmpeg volumedetect."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "volumedetect failed")
    return _parse_volumedetect(result.stderr)


def needs_normalization(
    stats: VolumeStats,
    *,
    mean_threshold: float = -22.0,
    max_threshold: float = -8.0,
) -> bool:
    return stats.mean_db < mean_threshold or stats.max_db < max_threshold


def optimize_audio_volume(
    input_path: str,
    output_path: str | None = None,
    *,
    in_place: bool = False,
    force: bool = False,
    mean_threshold: float = -22.0,
    max_threshold: float = -8.0,
    target_lufs: float = -16.0,
    true_peak_db: float = -1.5,
    lra: float = 11.0,
    bitrate: str = "192k",
) -> NormalizeResult:
    """Normalise quiet audio to podcast-friendly LUFS when thresholds are exceeded.

    Args:
        input_path: Source audio (typically cropped .m4a).
        output_path: Destination path; required unless ``in_place`` is True.
        in_place: Overwrite ``input_path`` when normalisation runs.
        force: Always apply loudnorm even when volume is already OK.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = measure_volume(str(src))
    should_norm = force or needs_normalization(
        stats, mean_threshold=mean_threshold, max_threshold=max_threshold
    )

    if in_place:
        dest = src
    elif output_path:
        dest = Path(output_path)
    else:
        raise ValueError("Provide output_path or set in_place=True")

    if not should_norm:
        if not in_place and dest.resolve() != src.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            dest.write_bytes(src.read_bytes())
        return NormalizeResult(
            path=str(dest),
            mean_db=stats.mean_db,
            max_db=stats.max_db,
            normalized=False,
            target_lufs=target_lufs,
            true_peak_db=true_peak_db,
        )

    tmp = dest.with_name(f"{dest.stem}.norm{dest.suffix}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    af = f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}"
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-i",
        str(src),
        "-af",
        af,
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "loudnorm failed")
    tmp.replace(dest)

    return NormalizeResult(
        path=str(dest),
        mean_db=stats.mean_db,
        max_db=stats.max_db,
        normalized=True,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
    )
