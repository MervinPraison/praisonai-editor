#!/usr/bin/env python3
"""Cut silent gaps from audio using ffmpeg silencedetect + concat (no speed change)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Defaults (tune via env vars if needed):
#   CUT_SILENCE_NOISE_DB=-30   peak-based silence threshold
#   CUT_SILENCE_MIN=1.5        only remove pauses >= 1.5 seconds
#   CUT_SILENCE_MARGIN=0.3     padding kept before/after each cut
NOISE_DB = float(os.getenv("CUT_SILENCE_NOISE_DB", "-30"))
MIN_SILENCE = float(os.getenv("CUT_SILENCE_MIN", "1.5"))
MARGIN = float(os.getenv("CUT_SILENCE_MARGIN", "0.3"))


def find_ffmpeg() -> str:
    for name in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        path = shutil.which(name) if "/" not in name else name
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def probe_duration(ffmpeg: str, path: str) -> float:
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    if not ffprobe.exists():
        ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    out = subprocess.check_output(
        [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        text=True,
    ).strip()
    return float(out)


def detect_silences(ffmpeg: str, path: str, noise_db: float, min_silence: float) -> list[tuple[float, float]]:
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stderr

    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in log.splitlines():
        if m := re.search(r"silence_start:\s*([\d.]+)", line):
            start = float(m.group(1))
        elif m := re.search(r"silence_end:\s*([\d.]+)", line):
            if start is not None:
                end = float(m.group(1))
                if end - start >= min_silence:
                    silences.append((start, end))
                start = None
    return silences


def keep_segments(duration: float, silences: list[tuple[float, float]], margin: float) -> list[tuple[float, float]]:
    if not silences:
        return [(0.0, duration)]

    segments: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in silences:
        end = s - margin
        if end > pos + 0.01:
            segments.append((pos, end))
        pos = e + margin

    if duration > pos + 0.01:
        segments.append((pos, duration))
    return segments


def render_segments(ffmpeg: str, src: Path, dst: Path, segments: list[tuple[float, float]]) -> None:
    ext = src.suffix.lower()
    if ext == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif ext in {".m4a", ".aac"}:
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = ["-c:a", "pcm_s16le"]

    if len(segments) == 1:
        start, end = segments[0]
        cmd = [ffmpeg, "-y", "-nostdin", "-ss", str(start), "-i", str(src),
               "-t", str(end - start), *codec, str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        return

    with tempfile.TemporaryDirectory(prefix="cut-silence-") as tmp:
        tmp_path = Path(tmp)
        parts: list[Path] = []
        for i, (start, end) in enumerate(segments):
            part = tmp_path / f"part_{i:04d}{ext if ext in {'.wav', '.mp3', '.m4a'} else '.wav'}"
            cmd = [ffmpeg, "-y", "-nostdin", "-ss", str(start), "-i", str(src),
                   "-t", str(end - start), *codec, str(part)]
            subprocess.run(cmd, check=True, capture_output=True)
            parts.append(part)

        list_file = tmp_path / "concat.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in parts) + "\n")
        cmd = [ffmpeg, "-y", "-nostdin", "-f", "concat", "-safe", "0",
               "-i", str(list_file), *codec, str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)


def cut_silence(input_path: str, output_path: str | None = None) -> str:
    ffmpeg = find_ffmpeg()
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_ALTERED{src.suffix}")

    duration = probe_duration(ffmpeg, str(src))
    silences = detect_silences(ffmpeg, str(src), NOISE_DB, MIN_SILENCE)
    segments = keep_segments(duration, silences, MARGIN)

    if len(segments) == 1 and segments[0][0] <= 0.01 and segments[0][1] >= duration - 0.01:
        shutil.copy2(src, dst)
    else:
        render_segments(ffmpeg, src, dst, segments)

    kept = sum(e - s for s, e in segments)
    removed = max(0.0, duration - kept)
    print(f"✓ {dst.name}: {duration:.1f}s → {kept:.1f}s (removed {removed:.1f}s, {100 * removed / duration:.1f}%)")
    return str(dst)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: cut-silence.py FILE [OUTPUT]", file=sys.stderr)
        return 1
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        cut_silence(inp, out)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
