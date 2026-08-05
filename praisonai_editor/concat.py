"""Concatenate multiple audio files into one via ffmpeg.

Usage:
    from praisonai_editor.concat import concat_audio

    # Same-codec inputs — lossless concat demuxer (stream copy)
    concat_audio(["part1.m4a", "part2.m4a"], "joined.m4a")

    # Mixed codecs / sample rates — re-encode through the concat filter
    concat_audio(["intro.mp3", "sermon.m4a"], "joined.m4a", reencode=True)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence


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


def _run_ffmpeg(cmd: List[str], verbose: bool = False) -> None:
    """Run an ffmpeg command, raising RuntimeError with the stderr tail on failure."""
    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode() if getattr(result, "stderr", None) else ""
        raise RuntimeError(f"FFmpeg concat failed: {stderr[-800:]}")


def _write_concat_list(paths: Sequence[Path], list_file: Path) -> None:
    """Write a concat-demuxer list file with absolute, quote-escaped paths."""
    lines = []
    for p in paths:
        abs_path = str(p.resolve())
        escaped = abs_path.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_audio(
    inputs: Sequence[str],
    output_path: str,
    *,
    reencode: bool = False,
    bitrate: str = "192k",
    sample_rate: int = 48000,
    channels: int = 2,
    verbose: bool = False,
) -> str:
    """Concatenate audio files into a single output.

    With ``reencode=False`` the ffmpeg concat demuxer is used with stream copy —
    fast and lossless, but all inputs must share codec, sample rate and channel
    layout. With ``reencode=True`` each input is conformed to ``sample_rate`` /
    ``channels`` via the concat filter and re-encoded to AAC, which handles
    mixed inputs.

    Args:
        inputs: Ordered audio files to join (at least one).
        output_path: Destination file path.
        reencode: Re-encode via concat filter instead of stream copy.
        bitrate: AAC bitrate when re-encoding (e.g. "128k", "192k").
        sample_rate: Target sample rate when re-encoding.
        channels: Target channel count when re-encoding (1 or 2).
        verbose: Print ffmpeg progress.

    Returns:
        Path to the concatenated file.
    """
    if not inputs:
        raise ValueError("At least one input file is required")
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")

    paths = [Path(p) for p in inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _find_ffmpeg()

    if reencode:
        cmd = [ffmpeg, "-y", "-nostdin"]
        for p in paths:
            cmd += ["-i", str(p)]

        n = len(paths)
        layout = "stereo" if channels == 2 else "mono"
        conform_parts = []
        branch_labels = []
        for i in range(n):
            conform_parts.append(
                f"[{i}:a]aformat=sample_rates={sample_rate}:channel_layouts={layout}[a{i}]"
            )
            branch_labels.append(f"[a{i}]")
        filter_complex = (
            ";".join(conform_parts)
            + ";"
            + "".join(branch_labels)
            + f"concat=n={n}:v=0:a=1[out]"
        )

        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "aac",
            "-b:a", bitrate,
            str(out),
        ]
        _run_ffmpeg(cmd, verbose)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            list_file = Path(tmp_dir) / "concat_list.txt"
            _write_concat_list(paths, list_file)
            cmd = [
                ffmpeg, "-y", "-nostdin",
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                str(out),
            ]
            _run_ffmpeg(cmd, verbose)

    return str(out)
