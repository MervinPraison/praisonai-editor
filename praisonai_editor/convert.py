"""Media format conversion via ffmpeg."""

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


class FFmpegConverter:
    """Converts media formats using ffmpeg. Implements the Converter protocol."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        *,
        bitrate: str = "192k",
    ) -> str:
        """Convert a media file to another format.

        The output format is determined by the output_path extension.
        Supports MP4→MP3, WAV→MP3, and other ffmpeg-supported conversions.

        Args:
            input_path: Path to source file
            output_path: Path for converted file
            bitrate: Audio bitrate (e.g. "128k", "192k", "320k")

        Returns:
            Path to the converted file
        """
        in_file = Path(input_path)
        if not in_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg = _find_ffmpeg()
        out_ext = out_file.suffix.lower()

        if out_ext == ".mp3":
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                "-vn",  # No video
                "-acodec", "libmp3lame",
                "-ab", bitrate,
                str(out_file),
            ]
        elif out_ext == ".wav":
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(out_file),
            ]
        elif out_ext == ".m4a":
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                "-vn",
                "-acodec", "aac",
                "-ab", bitrate,
                str(out_file),
            ]
        else:
            # Generic conversion
            cmd = [
                ffmpeg, "-y",
                "-i", str(in_file),
                str(out_file),
            ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else ""
            raise RuntimeError(f"FFmpeg conversion failed: {stderr}")

        return str(out_file)


# Module-level convenience function
_default_converter = None


def convert_media(
    input_path: str,
    output_path: str,
    *,
    bitrate: str = "192k",
) -> str:
    """Convert a media file to another format.

    Args:
        input_path: Path to source file
        output_path: Path for converted file (format determined by extension)
        bitrate: Audio bitrate

    Returns:
        Path to the converted file
    """
    global _default_converter
    if _default_converter is None:
        _default_converter = FFmpegConverter()
    return _default_converter.convert(input_path, output_path, bitrate=bitrate)
