"""Media file probing via ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .models import ProbeResult


def _find_ffprobe() -> str:
    """Find ffprobe executable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    for path in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe"]:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "ffprobe not found. Install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
    )


class FFmpegProber:
    """Probes media files using ffprobe. Implements the Prober protocol."""

    def probe(self, path: str) -> ProbeResult:
        """Probe a media file and return metadata."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ffprobe = _find_ffprobe()
        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")

        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        format_info = data.get("format", {})

        # Parse FPS from video stream
        fps = 0.0
        if video_stream:
            fps_str = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0

        return ProbeResult(
            path=str(file_path.absolute()),
            duration=float(format_info.get("duration", 0)),
            has_video=video_stream is not None,
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            audio_sample_rate=int(audio_stream.get("sample_rate", 0)) if audio_stream else None,
            audio_channels=int(audio_stream.get("channels", 0)) if audio_stream else None,
            audio_bitrate=int(audio_stream.get("bit_rate", 0)) if audio_stream and audio_stream.get("bit_rate") else None,
            video_codec=video_stream.get("codec_name") if video_stream else None,
            width=int(video_stream.get("width", 0)) if video_stream else 0,
            height=int(video_stream.get("height", 0)) if video_stream else 0,
            fps=fps,
            format_name=format_info.get("format_name", ""),
            size_bytes=int(format_info.get("size", 0)),
            bit_rate=int(format_info.get("bit_rate", 0)),
            streams=data.get("streams", []),
        )


# Module-level convenience function
_default_prober = None


def probe_media(path: str) -> ProbeResult:
    """Probe a media file using the default FFmpeg prober.

    Args:
        path: Path to the media file

    Returns:
        ProbeResult with file metadata
    """
    global _default_prober
    if _default_prober is None:
        _default_prober = FFmpegProber()
    return _default_prober.probe(path)
