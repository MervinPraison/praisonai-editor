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


#: How long each declick fade is, in milliseconds -- applied at every
#: INTERNAL join seam _render_concat creates (never at the very first
#: segment's start or the very last segment's end, which are real
#: recording boundaries, not edit seams). Short enough to be inaudible as
#: a "fade" during continuous speech, long enough to smooth the sample-
#: value/slope discontinuity that a hard join otherwise leaves at the
#: seam -- the actual cause of the audible click at a cut point. This is
#: the standard "declick" fade duration used for this exact problem
#: (distinct from a musical crossfade between two different sources,
#: which needs much longer, e.g. hundreds of ms to seconds).
DECLICK_FADE_MS = 10


def _audio_encode_args(out_ext: str, bitrate: str = "192k") -> List[str]:
    """The right -c:a (and any codec-specific) args for `out_ext`.

    Mirrors convert.py's FFmpegConverter own extension -> codec mapping --
    picking a codec that doesn't actually match the container (e.g.
    encoding MP3 audio into a file named .wav) produces a technically
    playable-in-ffmpeg-tools-only file that many other players/browsers
    choke on. Re-verified live: before this existed, remove_time_ranges'
    own reencode=True path on a .wav input produced a real "RIFF ...
    MPEG Layer 3" file -- valid enough for ffprobe to read, but not a real
    WAV. Falls through to a plain re-encode with ffmpeg's own default
    codec choice for any extension not explicitly listed here, same
    posture as convert.py's own "generic conversion" fallback.
    """
    ext = out_ext.lower()
    if ext == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", bitrate]
    if ext == ".wav":
        return ["-c:a", "pcm_s16le"]
    if ext in (".m4a", ".aac"):
        return ["-c:a", "aac", "-b:a", bitrate]
    return []


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
        declick_ms: float = DECLICK_FADE_MS,
    ) -> str:
        """Render audio based on edit plan.

        declick_ms: A short fade applied at every INTERNAL join seam
            _render_concat creates, to eliminate the audible click a hard
            join otherwise leaves at each cut point (the sample-value/
            slope discontinuity where two segments meet). Forces that
            segment to be re-encoded regardless of `copy_codec` -- a
            filter cannot be applied during a stream copy -- but only the
            segments that actually touch a seam; a single-keep-segment
            plan (_render_single, no internal joins at all) is unaffected
            and still honors `copy_codec` as before. Set to 0 to disable
            (restores the pre-fix behavior).
        """
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
            keep_segments, copy_codec, verbose, declick_ms,
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
            cmd.extend(_audio_encode_args(Path(output_path).suffix) or ["-c:a", "libmp3lame", "-b:a", "192k"])

        cmd.append(output_path)
        _run_ffmpeg(cmd, verbose)
        return output_path

    def _render_concat(
        self, ffmpeg: str, input_path: str, output_path: str,
        segments: List[Segment], copy_codec: bool, verbose: bool,
        declick_ms: float = DECLICK_FADE_MS,
    ) -> str:
        out_ext = Path(output_path).suffix
        fade_s = max(0.0, declick_ms) / 1000.0
        encode_args = _audio_encode_args(out_ext) or ["-c:a", "libmp3lame", "-b:a", "192k"]
        n = len(segments)
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

                # A hard join at an arbitrary sample almost never lines up
                # in value AND slope with the segment on the other side of
                # the cut -- that mismatch is what the ear hears as a
                # click. A brief fade across each INTERNAL seam smooths it
                # out. Never applied to the very first segment's start or
                # the very last segment's end: those are real recording
                # boundaries the edit never touched, not cut points.
                fade_filters = []
                if fade_s > 0:
                    if i > 0:
                        fade_filters.append(f"afade=t=in:st=0:d={fade_s}")
                    if i < n - 1:
                        fade_out_start = max(0.0, duration - fade_s)
                        fade_filters.append(f"afade=t=out:st={fade_out_start}:d={fade_s}")

                if fade_filters:
                    # A filter cannot run during a stream copy -- this
                    # segment touches a seam, so it must be decoded and
                    # re-encoded regardless of `copy_codec`.
                    cmd.extend(["-af", ",".join(fade_filters)])
                    cmd.extend(encode_args)
                elif copy_codec:
                    cmd.extend(["-c", "copy"])
                else:
                    cmd.extend(encode_args)

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
