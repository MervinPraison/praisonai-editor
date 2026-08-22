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
#   CUT_SILENCE_NOISE_DB=-30      peak-based silence threshold
#   CUT_SILENCE_MIN=1.5           only remove pauses >= 1.5 seconds
#   CUT_SILENCE_MARGIN=0.3        minimum keep at each silence boundary
#   CUT_SILENCE_STOP_LEAD=1.5     max seconds kept after silence_start (sentence tail)
#   CUT_SILENCE_STOP_RATIO=0.25   fraction of detected silence kept at start
#   CUT_SILENCE_RESUME_LEAD=1.5   max seconds kept before silence_end (quiet onset)
#   CUT_SILENCE_RESUME_RATIO=0.25 fraction of detected silence kept at tail
#   CUT_SILENCE_MIN_CORE=0.4      minimum seconds removed from each detected gap
NOISE_DB = float(os.getenv("CUT_SILENCE_NOISE_DB", "-30"))
MIN_SILENCE = float(os.getenv("CUT_SILENCE_MIN", "1.5"))
MARGIN = float(os.getenv("CUT_SILENCE_MARGIN", "0.3"))
STOP_LEAD = float(os.getenv("CUT_SILENCE_STOP_LEAD", "1.5"))
STOP_RATIO = float(os.getenv("CUT_SILENCE_STOP_RATIO", "0.25"))
RESUME_LEAD = float(os.getenv("CUT_SILENCE_RESUME_LEAD", "1.5"))
RESUME_RATIO = float(os.getenv("CUT_SILENCE_RESUME_RATIO", "0.25"))
MIN_CORE = float(os.getenv("CUT_SILENCE_MIN_CORE", "0.4"))


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


def _boundary_keep(silence_len: float, margin: float, lead: float, ratio: float) -> float:
    """Seconds to keep at one end of a detected silence (speech tail or quiet onset)."""
    return min(lead, max(margin, silence_len * ratio))


def _scaled_boundaries(
    silence_len: float,
    margin: float,
    stop_lead: float,
    stop_ratio: float,
    resume_lead: float,
    resume_ratio: float,
    min_core: float,
) -> tuple[float, float]:
    """Head/tail keep for speech edges; scale down if they would swallow the whole gap."""
    head = _boundary_keep(silence_len, margin, stop_lead, stop_ratio)
    tail = _boundary_keep(silence_len, margin, resume_lead, resume_ratio)
    max_keep = max(margin * 2, silence_len - min_core)
    if head + tail > max_keep:
        scale = max_keep / (head + tail)
        head *= scale
        tail *= scale
    return head, tail


def keep_segments(
    duration: float,
    silences: list[tuple[float, float]],
    margin: float,
    stop_lead: float = STOP_LEAD,
    stop_ratio: float = STOP_RATIO,
    resume_lead: float = RESUME_LEAD,
    resume_ratio: float = RESUME_RATIO,
    min_core: float = MIN_CORE,
) -> list[tuple[float, float]]:
    if not silences:
        return [(0.0, duration)]

    segments: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in silences:
        silence_len = e - s
        head, tail = _scaled_boundaries(
            silence_len, margin, stop_lead, stop_ratio, resume_lead, resume_ratio, min_core,
        )

        stop_at = s + head
        resume_at = e - tail

        if resume_at > stop_at + 0.01:
            if stop_at > pos + 0.01:
                segments.append((pos, stop_at))
            pos = max(pos, resume_at)

    if duration > pos + 0.01:
        segments.append((pos, duration))
    return segments


def _run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    subprocess.run([ffmpeg, *args], check=True, capture_output=True)


def _extract_segment(
    ffmpeg: str, src: Path, start: float, end: float, out: Path, codec: list[str],
) -> None:
    _run_ffmpeg(ffmpeg, [
        "-y", "-nostdin", "-ss", str(start), "-i", str(src),
        "-t", str(end - start), *codec, str(out),
    ])


def _probe_audio_format(ffmpeg: str, path: str) -> tuple[int, int]:
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    if not ffprobe.exists():
        ffprobe = Path(shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe")
    out = subprocess.check_output(
        [str(ffprobe), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        text=True,
    ).strip().splitlines()
    return int(out[0]), int(out[1])


def render_segments(
    ffmpeg: str,
    src: Path,
    dst: Path,
    segments: list[tuple[float, float]],
    extract_src: Path | None = None,
    source_offset: float = 0.0,
) -> None:
    ext = src.suffix.lower()
    if ext == ".mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif ext in {".m4a", ".aac"}:
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = ["-c:a", "pcm_s16le"]

    read_src = extract_src or src

    if len(segments) == 1:
        start, end = segments[0]
        _extract_segment(ffmpeg, read_src, start + source_offset, end + source_offset, dst, codec)
        return

    sample_rate, channels = _probe_audio_format(ffmpeg, str(read_src))
    with tempfile.TemporaryDirectory(prefix="cut-silence-", dir=dst.parent) as tmp:
        tmp_path = Path(tmp)
        raw_path = tmp_path / "stream.pcm"
        part_path = tmp_path / "part.pcm"

        for i, (start, end) in enumerate(segments):
            _run_ffmpeg(ffmpeg, [
                "-y", "-nostdin", "-ss", str(start + source_offset),
                "-i", str(read_src), "-t", str(end - start),
                "-f", "s16le", "-acodec", "pcm_s16le", str(part_path),
            ])
            mode = "wb" if i == 0 else "ab"
            with open(raw_path, mode) as out_f, open(part_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)
            part_path.unlink(missing_ok=True)

        _run_ffmpeg(ffmpeg, [
            "-y", "-nostdin", "-f", "s16le",
            "-ar", str(sample_rate), "-ac", str(channels),
            "-i", str(raw_path), *codec, str(dst),
        ])


def cut_silence(input_path: str, output_path: str | None = None) -> str:
    ffmpeg = find_ffmpeg()
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_ALTERED{src.suffix}")

    duration = probe_duration(ffmpeg, str(src))
    silences = detect_silences(ffmpeg, str(src), NOISE_DB, MIN_SILENCE)
    segments = keep_segments(duration, silences, MARGIN)

    extract_src_path = Path(os.getenv("CUT_SILENCE_EXTRACT_SRC", str(src)))
    source_offset = float(os.getenv("CUT_SILENCE_SOURCE_OFFSET", "0"))

    if len(segments) == 1 and segments[0][0] <= 0.01 and segments[0][1] >= duration - 0.01:
        shutil.copy2(src, dst)
    else:
        render_segments(
            ffmpeg, src, dst, segments,
            extract_src=extract_src_path if extract_src_path != src else None,
            source_offset=source_offset,
        )

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
