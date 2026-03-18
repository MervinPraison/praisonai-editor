"""Media rendering using FFmpeg — audio and video."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from .models import EditPlan, Segment


def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def _run_ffmpeg(cmd: List[str], verbose: bool = False) -> None:
    """Run FFmpeg command."""
    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode() if hasattr(result, "stderr") and result.stderr else ""
        raise RuntimeError(f"FFmpeg failed: {stderr}")


class FFmpegAudioRenderer:
    """Renders edited audio using FFmpeg. Implements the Renderer protocol."""

    def render(
        self,
        input_path: str,
        output_path: str,
        plan: EditPlan,
        *,
        copy_codec: bool = True,
        verbose: bool = False,
    ) -> str:
        """Render audio based on edit plan."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ffmpeg = _find_ffmpeg()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        keep_segments = plan.get_keep_segments()
        if not keep_segments:
            raise ValueError("No segments to keep in edit plan")

        if len(keep_segments) == 1:
            return self._render_single(
                ffmpeg, str(input_file), str(output_file),
                keep_segments[0], copy_codec, verbose,
            )

        return self._render_concat(
            ffmpeg, str(input_file), str(output_file),
            keep_segments, copy_codec, verbose,
        )

    def _render_single(
        self, ffmpeg: str, input_path: str, output_path: str,
        seg: Segment, copy_codec: bool, verbose: bool,
    ) -> str:
        duration = seg.end - seg.start
        cmd = [ffmpeg, "-y"]

        if copy_codec:
            cmd.extend(["-ss", str(seg.start)])
            cmd.extend(["-i", input_path])
            cmd.extend(["-t", str(duration)])
            cmd.extend(["-c", "copy"])
        else:
            cmd.extend(["-i", input_path])
            cmd.extend(["-ss", str(seg.start)])
            cmd.extend(["-t", str(duration)])
            cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])

        cmd.append(output_path)
        _run_ffmpeg(cmd, verbose)
        return output_path

    def _render_concat(
        self, ffmpeg: str, input_path: str, output_path: str,
        segments: List[Segment], copy_codec: bool, verbose: bool,
    ) -> str:
        out_ext = Path(output_path).suffix
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_files = []
            for i, seg in enumerate(segments):
                seg_path = Path(tmpdir) / f"seg_{i:04d}{out_ext}"
                duration = seg.end - seg.start
                cmd = [
                    ffmpeg, "-y",
                    "-ss", str(seg.start),
                    "-i", input_path,
                    "-t", str(duration),
                ]
                if copy_codec:
                    cmd.extend(["-c", "copy"])
                else:
                    cmd.extend(["-c:a", "libmp3lame", "-b:a", "192k"])
                cmd.append(str(seg_path))
                _run_ffmpeg(cmd, verbose)
                segment_files.append(seg_path)

            concat_file = Path(tmpdir) / "concat.txt"
            with open(concat_file, "w") as f:
                for seg_path in segment_files:
                    f.write(f"file '{seg_path}'\n")

            cmd = [
                ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                output_path,
            ]
            _run_ffmpeg(cmd, verbose)
        return output_path


class FFmpegVideoRenderer:
    """Renders edited video using FFmpeg. Implements the Renderer protocol."""

    def render(
        self,
        input_path: str,
        output_path: str,
        plan: EditPlan,
        *,
        copy_codec: bool = True,
        verbose: bool = False,
    ) -> str:
        """Render video based on edit plan."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        ffmpeg = _find_ffmpeg()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        keep_segments = plan.get_keep_segments()
        if not keep_segments:
            raise ValueError("No segments to keep in edit plan")

        if len(keep_segments) == 1:
            return self._render_single(
                ffmpeg, str(input_file), str(output_file),
                keep_segments[0], copy_codec, verbose,
            )

        return self._render_concat(
            ffmpeg, str(input_file), str(output_file),
            keep_segments, copy_codec, verbose,
        )

    def _render_single(
        self, ffmpeg: str, input_path: str, output_path: str,
        seg: Segment, copy_codec: bool, verbose: bool,
    ) -> str:
        duration = seg.end - seg.start
        cmd = [ffmpeg, "-y"]
        if copy_codec:
            cmd.extend(["-ss", str(seg.start)])
            cmd.extend(["-i", input_path])
            cmd.extend(["-t", str(duration)])
            cmd.extend(["-c", "copy"])
        else:
            cmd.extend(["-i", input_path])
            cmd.extend(["-ss", str(seg.start)])
            cmd.extend(["-t", str(duration)])
            cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        cmd.append(output_path)
        _run_ffmpeg(cmd, verbose)
        return output_path

    def _render_concat(
        self, ffmpeg: str, input_path: str, output_path: str,
        segments: List[Segment], copy_codec: bool, verbose: bool,
    ) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            segment_files = []
            for i, seg in enumerate(segments):
                seg_path = Path(tmpdir) / f"seg_{i:04d}.mp4"
                duration = seg.end - seg.start
                cmd = [
                    ffmpeg, "-y",
                    "-ss", str(seg.start),
                    "-i", input_path,
                    "-t", str(duration),
                ]
                if copy_codec:
                    cmd.extend(["-c", "copy"])
                else:
                    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
                    cmd.extend(["-c:a", "aac", "-b:a", "128k"])
                cmd.append(str(seg_path))
                _run_ffmpeg(cmd, verbose)
                segment_files.append(seg_path)

            concat_file = Path(tmpdir) / "concat.txt"
            with open(concat_file, "w") as f:
                for seg_path in segment_files:
                    f.write(f"file '{seg_path}'\n")

            cmd = [
                ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                output_path,
            ]
            _run_ffmpeg(cmd, verbose)
        return output_path
