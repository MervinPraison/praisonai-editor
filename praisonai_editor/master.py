"""Two-pass EBU R128 loudness mastering for streaming targets (YouTube -14 LUFS).

Pass 1 measures integrated loudness / true peak / loudness range with
``loudnorm=print_format=json``; pass 2 applies loudnorm in *linear* mode
(one constant gain — no pumping) with all ``measured_*`` values, preceded by
a preset compressor and followed by a true-peak safety limiter and a
resample back to the delivery rate (loudnorm upsamples to 192 kHz
internally).

Usage:
    from praisonai_editor.master import master_audio, measure_loudness

    result = master_audio("sermon.m4a")                 # speech preset, -14 LUFS
    result = master_audio("song.m4a", preset="music")
    stats = measure_loudness("sermon.m4a")              # LoudnessStats
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .convert import _find_ffmpeg

#: Preset upstream processing (applied BEFORE loudnorm) and loudness-range target.
#: speech: firmer 3:1 compression for even sermon/podcast delivery (LRA 11).
#: music: gentler 2:1 compression that preserves dynamics (LRA 15).
MASTER_PRESETS: dict[str, dict] = {
    "speech": {
        "pre_chain": ["acompressor=threshold=-18dB:ratio=3:attack=20:release=250:makeup=4dB"],
        "lra": 11.0,
    },
    "music": {
        "pre_chain": ["acompressor=threshold=-16dB:ratio=2:attack=25:release=300:makeup=2dB"],
        "lra": 15.0,
    },
}

#: -1 dBTP ceiling (10 ** (-1 / 20) ≈ 0.891) as a safety net after loudnorm.
_LIMITER = "alimiter=limit=0.891:level=false"

#: Auto-preset boundary: midpoint of the speech (11) and music (15) LRA targets.
_AUTO_LRA_MUSIC_THRESHOLD = 13.0


@dataclass(frozen=True)
class LoudnessStats:
    """EBU R128 measurement from a loudnorm analysis pass."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


@dataclass(frozen=True)
class MasterResult:
    path: str
    stats: LoudnessStats
    preset: str
    chain: str
    target_lufs: float
    true_peak_db: float
    normalized: bool = True


def _to_float(value) -> float:
    """Parse a loudnorm JSON value (strings, possibly '-inf') to float."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"loudnorm stat not a number: {value!r}") from exc


def _parse_loudnorm_json(stderr: str) -> LoudnessStats:
    """Extract the LAST loudnorm JSON block from ffmpeg stderr."""
    blocks = re.findall(r"\{[^{}]*\}", stderr)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "input_i" in data:
            return LoudnessStats(
                input_i=_to_float(data["input_i"]),
                input_tp=_to_float(data["input_tp"]),
                input_lra=_to_float(data["input_lra"]),
                input_thresh=_to_float(data["input_thresh"]),
                target_offset=_to_float(data.get("target_offset", 0.0)),
            )
    raise RuntimeError(f"loudnorm measurement parse failed: {stderr[-800:]}")


def measure_loudness(
    path: str, verbose: bool = False, pre_chain: list[str] | None = None
) -> LoudnessStats:
    """Measure integrated loudness / true peak / LRA via a loudnorm analysis pass.

    ``pre_chain`` must list any filters that will run BEFORE loudnorm in
    the apply pass. Loudnorm's ``measured_*`` values only produce an
    on-target result when they describe the very signal loudnorm will
    receive — measuring the raw input while the apply pass first pushes
    it through a compressor's makeup gain lands the output above target
    by exactly that gain.
    """
    ffmpeg = _find_ffmpeg()
    analysis = list(pre_chain or []) + [
        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json"
    ]
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-af",
        ",".join(analysis),
        "-f",
        "null",
        "-",
    ]
    if verbose:
        print(f"Measuring loudness: {path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "loudnorm measurement failed")
    return _parse_loudnorm_json(result.stderr)


def _pick_preset(stats: LoudnessStats) -> str:
    """Choose speech vs music from measured stats (for preset='auto').

    Heuristic: speech is naturally narrow in loudness range (the speech
    preset itself targets LRA 11) while music/worship recordings measure
    wider dynamics. Inputs at or above the midpoint of the two preset LRA
    targets (13 LU) are treated as music; everything else as speech.
    """
    return "music" if stats.input_lra >= _AUTO_LRA_MUSIC_THRESHOLD else "speech"


def _build_master_filter(
    pre_chain: list[str],
    stats: LoudnessStats,
    target_lufs: float,
    true_peak_db: float,
    lra: float,
    sample_rate: int,
) -> str:
    """Build the pass-2 -af chain: pre chain + linear loudnorm + limiter + resample."""
    loudnorm = (
        f"loudnorm=I={target_lufs:g}:TP={true_peak_db:g}:LRA={lra:g}"
        f":measured_I={stats.input_i:g}:measured_TP={stats.input_tp:g}"
        f":measured_LRA={stats.input_lra:g}:measured_thresh={stats.input_thresh:g}"
        f":offset={stats.target_offset:g}:linear=true"
    )
    filters = list(pre_chain) + [loudnorm, _LIMITER, f"aresample={sample_rate}"]
    return ",".join(filters)


def master_audio(
    input_path: str,
    output_path: str | None = None,
    *,
    preset: str = "speech",
    target_lufs: float = -14.0,
    true_peak_db: float = -1.5,
    lra: float | None = None,
    chain: list[str] | None = None,
    sample_rate: int = 48000,
    channels: int = 2,
    bitrate: str = "192k",
    verbose: bool = False,
) -> MasterResult:
    """Master audio to a streaming loudness target (two-pass EBU R128 loudnorm).

    Args:
        input_path: Source audio file.
        output_path: Destination path (default: ``{stem}.mastered.m4a`` next
            to the input).
        preset: "speech", "music", or "auto" (pick from measured stats).
        target_lufs: Integrated loudness target (default -14, YouTube norm).
        true_peak_db: True-peak ceiling in dBTP (default -1.5).
        lra: Loudness-range target; default comes from the preset.
        chain: Custom upstream filter list; fully REPLACES the preset
            pre-chain (loudnorm + limiter + resample are always appended).
            Pass ``[]`` for loudness-only mastering with no compression.
        sample_rate: Output sample rate in Hz.
        channels: Output channels — 1 (mono) or 2 (stereo).
        bitrate: AAC bitrate (e.g. "128k", "192k").
        verbose: Print ffmpeg progress.

    Returns:
        MasterResult with the pass-1 stats and the applied filter chain.
        Silent inputs (measured integrated loudness of -inf) are transcoded
        without loudness normalisation and flagged ``normalized=False``.
    """
    if preset not in ("speech", "music", "auto"):
        raise ValueError(f"preset must be speech, music, or auto, got {preset!r}")
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out = Path(output_path) if output_path else src.parent / f"{src.stem}.mastered.m4a"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: measure the raw input — the stats we report, the silence
    # test, and the auto-preset decision all describe the source itself.
    stats = measure_loudness(str(src), verbose=verbose)

    resolved_preset = _pick_preset(stats) if preset == "auto" else preset
    spec = MASTER_PRESETS[resolved_preset]
    effective_lra = lra if lra is not None else float(spec["lra"])

    # Pass 2: apply. Silence (input_i == -inf) has no loudness to normalise —
    # loudnorm would blow up its gain, so transcode/resample only.
    silent = stats.input_i == float("-inf")
    if silent:
        af = f"aresample={sample_rate}"
    else:
        pre_chain = list(chain) if chain is not None else list(spec["pre_chain"])
        # loudnorm sees the POST-pre-chain signal, so that is what its
        # measured_* values must describe — otherwise the compressor's
        # makeup gain rides on top of loudnorm's constant gain and the
        # output overshoots the target by exactly that much. Re-measure
        # through the pre-chain (skipped when there is none).
        apply_stats = (
            measure_loudness(str(src), verbose=verbose, pre_chain=pre_chain)
            if pre_chain
            else stats
        )
        af = _build_master_filter(
            pre_chain,
            apply_stats,
            target_lufs,
            true_peak_db,
            effective_lra,
            sample_rate,
        )

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-nostdin",
        "-i", str(src),
        "-af", af,
        "-ac", str(channels),
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
        raise RuntimeError(f"FFmpeg master failed: {stderr[-800:]}")

    return MasterResult(
        path=str(out),
        stats=stats,
        preset=resolved_preset,
        chain=af,
        target_lufs=target_lufs,
        true_peak_db=true_peak_db,
        normalized=not silent,
    )
