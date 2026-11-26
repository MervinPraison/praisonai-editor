"""Background-noise QA: measure room tone and recommend afftdn settings.

Uses ffmpeg ``astats`` per-window sampling (RMS, peak, noise floor, RMS
trough) plus ``silencedetect`` and ``volumedetect``.  Based on broadcast
practice: measure quiet regions first, set ``afftdn`` ``nf`` near the
measured floor, and keep ``nr`` in the 10–20 dB range to avoid artefacts.

Usage:
    from praisonai_editor.noise_qa import analyse_noise, apply_noise_qa

    report = analyse_noise("recording.m4a")
    result = apply_noise_qa("recording.m4a")  # QA + conditional denoise
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .convert import _find_ffmpeg
from .denoise import denoise_audio
from .normalize import measure_volume


@dataclass(frozen=True)
class NoiseWindow:
    time: float
    rms_db: float
    peak_db: float
    noise_floor_db: float | None
    rms_trough_db: float | None


@dataclass
class NoiseQAReport:
    """Background-noise analysis and denoise recommendation."""

    input_path: str
    duration_sec: float = 0.0
    mean_volume_db: float = 0.0
    max_volume_db: float = 0.0
    median_rms_db: float = 0.0
    quiet_rms_db: float = 0.0
    noise_floor_db: float = -60.0
    snr_db: float = 0.0
    silence_ratio: float = 0.0
    level: str = "clean"  # clean | mild | moderate | heavy
    apply_denoise: bool = False
    noise_reduction: float = 0.0
    noise_floor_param: float = -50.0
    track_noise: bool = True
    highpass_hz: int | None = None
    reason: str = ""
    windows: list[NoiseWindow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["windows"] = [asdict(w) for w in self.windows]
        return d

    def summary(self) -> str:
        lines = [
            f"Noise QA: {self.level.upper()}",
            f"  Estimated noise floor: {self.noise_floor_db:.1f} dB",
            f"  Content RMS (median):  {self.median_rms_db:.1f} dB",
            f"  SNR proxy:             {self.snr_db:.1f} dB",
            f"  Mean / peak volume:    {self.mean_volume_db:.1f} / {self.max_volume_db:.1f} dB",
            f"  Silence ratio:         {self.silence_ratio:.0%}",
            f"  Recommendation:        {self.reason}",
        ]
        if self.apply_denoise:
            hp = f", highpass {self.highpass_hz} Hz" if self.highpass_hz else ""
            lines.append(
                f"  Denoise: afftdn nr={self.noise_reduction:g} nf={self.noise_floor_param:g}"
                f" tn={1 if self.track_noise else 0}{hp}"
            )
        else:
            lines.append("  Denoise: skip (clean enough)")
        return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return -60.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * pct / 100)))
    return ordered[idx]


def _probe_duration(ffmpeg: str, path: str) -> float:
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    out = subprocess.check_output(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).strip()
    return float(out)


def _sample_windows(
    ffmpeg: str,
    path: str,
    *,
    window_sec: float = 0.5,
    sample_rate: int = 44100,
) -> list[NoiseWindow]:
    """Per-window astats via ametadata log (see mac/audio-db-chart.py)."""
    reset = max(1, int(sample_rate * window_sec))
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
        log_path = tmp.name

    af = (
        f"asetnsamples={reset},"
        "astats=metadata=1:reset=1:"
        "measure_overall=RMS_level+Peak_level+Noise_floor+RMS_trough,"
        f"ametadata=print:file={log_path}"
    )
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-i", path, "-af", af, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(log_path).unlink(missing_ok=True)
        raise RuntimeError(f"astats window sampling failed: {result.stderr[-800:]}")

    windows: list[NoiseWindow] = []
    cur_time = 0.0
    cur: dict[str, float | None] = {}

    def _flush() -> None:
        nonlocal cur, cur_time
        if cur.get("rms") is None:
            return
        windows.append(
            NoiseWindow(
                time=cur_time,
                rms_db=float(cur["rms"]),  # type: ignore[arg-type]
                peak_db=float(cur.get("peak") or cur["rms"]),  # type: ignore[arg-type]
                noise_floor_db=cur.get("nf"),  # type: ignore[arg-type]
                rms_trough_db=cur.get("trough"),  # type: ignore[arg-type]
            )
        )

    for line in Path(log_path).read_text().splitlines():
        if m := re.search(r"pts_time:([\d.]+)", line):
            _flush()
            cur_time = float(m.group(1))
            cur = {}
        elif "Overall.RMS_level=" in line:
            val = line.split("=")[-1].strip()
            cur["rms"] = -100.0 if val in {"-inf", "inf", "nan"} else float(val)
        elif "Overall.Peak_level=" in line:
            val = line.split("=")[-1].strip()
            cur["peak"] = -100.0 if val in {"-inf", "inf", "nan"} else float(val)
        elif "Overall.Noise_floor=" in line:
            val = line.split("=")[-1].strip()
            cur["nf"] = None if val in {"-inf", "inf", "nan"} else float(val)
        elif "Overall.RMS_trough=" in line:
            val = line.split("=")[-1].strip()
            cur["trough"] = -100.0 if val in {"-inf", "inf", "nan"} else float(val)

    _flush()
    Path(log_path).unlink(missing_ok=True)
    return windows


def _detect_silence_ratio(
    ffmpeg: str,
    path: str,
    *,
    noise_db: float = -30.0,
    min_silence: float = 0.3,
    duration: float,
) -> float:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    silent = 0.0
    start: float | None = None
    for line in result.stderr.splitlines():
        if m := re.search(r"silence_start:\s*([\d.]+)", line):
            start = float(m.group(1))
        elif m := re.search(r"silence_end:\s*([\d.]+)", line):
            if start is not None:
                silent += float(m.group(1)) - start
                start = None
    if duration > 0:
        return min(1.0, silent / duration)
    return 0.0


def _estimate_noise_floor(windows: list[NoiseWindow]) -> tuple[float, float, float]:
    """Return (noise_floor_db, median_rms_db, quiet_rms_db)."""
    if not windows:
        return -60.0, -30.0, -40.0

    rms_vals = [w.rms_db for w in windows]
    trough_vals = [w.rms_trough_db for w in windows if w.rms_trough_db is not None]
    nf_vals = [w.noise_floor_db for w in windows if w.noise_floor_db is not None]

    median_rms = statistics.median(rms_vals)
    quiet_rms = statistics.mean(sorted(rms_vals)[: max(1, len(rms_vals) // 10)])

    candidates: list[float] = []
    if trough_vals:
        candidates.append(_percentile(trough_vals, 10))
    if nf_vals:
        candidates.append(_percentile(nf_vals, 10))
    candidates.append(quiet_rms)

    # Prefer the quietest reliable estimate, capped to afftdn nf range.
    noise_floor = min(candidates)
    noise_floor = max(-80.0, min(-20.0, noise_floor))
    return noise_floor, median_rms, quiet_rms


def classify_noise(
    noise_floor_db: float,
    snr_db: float,
    *,
    silence_ratio: float = 0.0,
) -> tuple[str, bool, float, float, bool, int | None, str]:
    """Map metrics → level, apply flag, nr, nf, track_noise, highpass, reason."""
    # Continuous background (little silence but elevated floor) → treat as noisy.
    if silence_ratio < 0.05 and noise_floor_db > -42.0:
        snr_db = min(snr_db, (noise_floor_db + 50.0))  # penalise constant bed

    if noise_floor_db <= -52.0 and snr_db >= 28.0:
        return (
            "clean",
            False,
            0.0,
            -50.0,
            True,
            None,
            "Noise floor very low and good SNR — denoise not needed.",
        )

    if noise_floor_db <= -45.0 and snr_db >= 22.0:
        nr, nf = 10.0, min(-20.0, noise_floor_db + 8.0)
        return (
            "mild",
            True,
            nr,
            nf,
            True,
            None,
            "Light hiss/hum — gentle denoise recommended.",
        )

    # Mobile / continuous speech: no silent gaps for tn=1 to learn — match nf to floor.
    if silence_ratio < 0.05 and noise_floor_db > -42.0:
        nr = 28.0 if noise_floor_db > -38.0 else 24.0
        nf = min(-20.0, noise_floor_db + 2.0)
        return (
            "moderate",
            True,
            nr,
            nf,
            False,
            120,
            "Mobile/continuous env noise — static afftdn with nf matched to measured floor.",
        )

    if noise_floor_db <= -35.0 or snr_db >= 12.0:
        nr, nf = 14.0, min(-20.0, noise_floor_db + 5.0)
        hp = 80 if noise_floor_db > -40.0 else None
        return (
            "moderate",
            True,
            nr,
            nf,
            False if silence_ratio < 0.05 else True,
            hp,
            "Audible background bed — standard podcast denoise.",
        )

    nr, nf = 20.0, min(-20.0, noise_floor_db + 3.0)
    return (
        "heavy",
        True,
        nr,
        nf,
        True,
        80,
        "Strong constant noise — stronger denoise + high-pass.",
    )


def analyse_noise(
    input_path: str,
    *,
    window_sec: float = 0.5,
    silence_detect_db: float = -30.0,
) -> NoiseQAReport:
    """Analyse background noise and return a denoise recommendation."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ffmpeg = _find_ffmpeg()
    duration = _probe_duration(ffmpeg, str(src))
    vol = measure_volume(str(src))
    windows = _sample_windows(ffmpeg, str(src), window_sec=window_sec)
    noise_floor, median_rms, quiet_rms = _estimate_noise_floor(windows)
    snr = median_rms - noise_floor
    silence_ratio = _detect_silence_ratio(
        ffmpeg, str(src), noise_db=silence_detect_db, duration=duration
    )

    level, apply, nr, nf, tn, hp, reason = classify_noise(
        noise_floor, snr, silence_ratio=silence_ratio
    )

    return NoiseQAReport(
        input_path=str(src),
        duration_sec=duration,
        mean_volume_db=vol.mean_db,
        max_volume_db=vol.max_db,
        median_rms_db=median_rms,
        quiet_rms_db=quiet_rms,
        noise_floor_db=noise_floor,
        snr_db=snr,
        silence_ratio=silence_ratio,
        level=level,
        apply_denoise=apply,
        noise_reduction=nr,
        noise_floor_param=nf,
        track_noise=tn,
        highpass_hz=hp,
        reason=reason,
        windows=windows,
    )


def _remux_video_with_audio(
    ffmpeg: str,
    video_path: str,
    audio_path: str,
    output_path: str,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Remux failed: {result.stderr[-800:]}")


def apply_noise_qa(
    input_path: str,
    output_path: str | None = None,
    *,
    force_denoise: bool = False,
    skip_denoise: bool = False,
    verbose: bool = False,
    **analyse_kwargs: Any,
) -> tuple[NoiseQAReport, str | None]:
    """Run QA and optionally denoise (remuxes video when input has a video stream).

    Returns ``(report, output_path)`` — ``output_path`` is ``None`` when denoise
    was skipped and no file was written.
    """
    report = analyse_noise(input_path, **analyse_kwargs)
    src = Path(input_path)
    should_denoise = report.apply_denoise and not skip_denoise
    if force_denoise:
        should_denoise = True

    if not should_denoise:
        return report, None

    ffmpeg = _find_ffmpeg()
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    has_video = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(src),
        ],
        capture_output=True,
        text=True,
    ).stdout.strip() == "video"

    stem = src.stem
    if output_path:
        out = Path(output_path)
    elif has_video:
        out = src.parent / f"{stem}_denoised{src.suffix}"
    else:
        out = src.parent / f"{stem}_denoised.m4a"

    # Optional high-pass before afftdn for hum/rumble.
    if report.highpass_hz:
        pre = src.parent / f"{stem}_hp.m4a"
        af = f"highpass=f={report.highpass_hz}"
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
            "192k",
            str(pre),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        denoise_src = str(pre)
    else:
        denoise_src = str(src)

    denoise_out = src.parent / f"{stem}_dn_audio.m4a"
    denoise_audio(
        denoise_src,
        str(denoise_out),
        noise_reduction=report.noise_reduction,
        noise_floor=report.noise_floor_param,
        track_noise=report.track_noise,
        verbose=verbose,
    )

    if report.highpass_hz and Path(denoise_src).exists() and denoise_src != str(src):
        Path(denoise_src).unlink(missing_ok=True)

    if has_video:
        _remux_video_with_audio(ffmpeg, str(src), str(denoise_out), str(out))
        denoise_out.unlink(missing_ok=True)
    else:
        denoise_out.rename(out)

    return report, str(out)


def write_report_json(report: NoiseQAReport, path: str) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
