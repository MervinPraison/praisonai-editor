#!/usr/bin/env python3
"""Plot audio loudness (dB) over time using ffmpeg astats."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Match cut-silence.py defaults — silencedetect uses peak level
CUT_NOISE_DB = -30
CUT_MIN_SILENCE = 1.5


def find_ffmpeg() -> str:
    for name in ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        path = shutil.which(name) if "/" not in name else name
        if path and Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def sample_rate(ffmpeg: str, path: str) -> int:
    ffprobe = Path(ffmpeg).with_name("ffprobe")
    if not ffprobe.exists():
        ffprobe = Path(shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe")
    out = subprocess.check_output(
        [str(ffprobe), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", path],
        text=True,
    ).strip()
    return int(float(out))


def extract_levels(ffmpeg: str, path: str, window_sec: float = 1.0) -> tuple[list[float], list[float], list[float]]:
    sr = sample_rate(ffmpeg, path)
    reset = max(1, int(sr * window_sec))
    log_file = Path(path).with_suffix(".db_stats.log")

    af = (
        f"asetnsamples={reset},"
        f"astats=metadata=1:reset=1:measure_overall=RMS_level+Peak_level,"
        f"ametadata=print:file={log_file}"
    )
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-i", path, "-af", af, "-f", "null", "-"],
        check=True,
        capture_output=True,
    )

    times: list[float] = []
    rms: list[float] = []
    peak: list[float] = []
    cur_time = 0.0
    cur_rms = None
    cur_peak = None

    for line in log_file.read_text().splitlines():
        if m := re.search(r"pts_time:([\d.]+)", line):
            if cur_rms is not None:
                times.append(cur_time)
                rms.append(cur_rms)
                peak.append(cur_peak if cur_peak is not None else cur_rms)
            cur_time = float(m.group(1))
            cur_rms = cur_peak = None
        elif "Overall.RMS_level=" in line:
            val = line.split("Overall.RMS_level=")[-1].strip()
            cur_rms = -100.0 if val in {"-inf", "inf"} else float(val)
        elif "Overall.Peak_level=" in line:
            val = line.split("Overall.Peak_level=")[-1].strip()
            cur_peak = -100.0 if val in {"-inf", "inf"} else float(val)

    if cur_rms is not None:
        times.append(cur_time)
        rms.append(cur_rms)
        peak.append(cur_peak if cur_peak is not None else cur_rms)

    log_file.unlink(missing_ok=True)
    if not times:
        raise RuntimeError("No audio levels extracted")
    return times, rms, peak


def _time_label(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def _silence_gaps(times: list[float], levels: list[float], threshold: float, min_duration: float) -> list[dict]:
    gaps: list[dict] = []
    start: float | None = None
    window = times[1] - times[0] if len(times) > 1 else 1.0

    for t, level in zip(times, levels):
        quiet = level <= threshold
        if quiet and start is None:
            start = t
        elif not quiet and start is not None:
            end = t
            if end - start >= min_duration:
                gaps.append({
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(end - start, 3),
                    "start_label": _time_label(start),
                    "end_label": _time_label(end),
                })
            start = None

    if start is not None and times:
        end = times[-1] + window
        if end - start >= min_duration:
            gaps.append({
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "start_label": _time_label(start),
                "end_label": _time_label(end),
            })
    return gaps


def build_report(path: str, times: list[float], rms: list[float], peak: list[float], window_sec: float) -> dict:
    audible = [v for v in peak if v > CUT_NOISE_DB]
    gaps = _silence_gaps(times, peak, CUT_NOISE_DB, CUT_MIN_SILENCE)
    return {
        "format": "praisonai-audio-db-chart/v1",
        "source_file": str(Path(path).resolve()),
        "duration_sec": round(times[-1], 3),
        "duration_label": _time_label(times[-1]),
        "window_sec": window_sec,
        "sample_count": len(times),
        "units": {"level": "dBFS", "time": "seconds"},
        "cut_silence_defaults": {
            "noise_db": CUT_NOISE_DB,
            "min_silence_sec": CUT_MIN_SILENCE,
            "description": "Audio with peak at or below noise_db for min_silence_sec is treated as silence (matches silencedetect).",
        },
        "summary": {
            "rms_min_db": round(min(rms), 2),
            "rms_max_db": round(max(rms), 2),
            "rms_mean_db": round(sum(rms) / len(rms), 2),
            "peak_min_db": round(min(peak), 2),
            "peak_max_db": round(max(peak), 2),
            "peak_above_cut_threshold_pct": round(100 * len(audible) / len(peak), 1),
            "silence_gap_count": len(gaps),
            "silence_gap_total_sec": round(sum(g["duration_sec"] for g in gaps), 1),
        },
        "silence_gaps": gaps,
        "samples": [
            {
                "time_sec": round(t, 3),
                "time_label": _time_label(t),
                "rms_db": round(r, 2),
                "peak_db": round(p, 2),
                "below_cut_threshold": p <= CUT_NOISE_DB,
            }
            for t, r, p in zip(times, rms, peak)
        ],
    }


def save_json(report: dict, out: Path) -> None:
    out.write_text(json.dumps(report, indent=2) + "\n")


def save_csv(times: list[float], rms: list[float], peak: list[float], out: Path) -> None:
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "time_label", "rms_db", "peak_db", "below_cut_threshold"])
        for t, r, p in zip(times, rms, peak):
            writer.writerow([round(t, 3), _time_label(t), round(r, 2), round(p, 2), p <= CUT_NOISE_DB])


def save_chart(path: str, times: list[float], rms: list[float], peak: list[float], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.plot(times, peak, color="#e45756", linewidth=0.6, alpha=0.7, label="Peak dB")
    ax.plot(times, rms, color="#4c78a8", linewidth=0.8, label="RMS dB")
    ax.axhline(CUT_NOISE_DB, color="#54a24b", linestyle="--", linewidth=1.2,
               label=f"Peak cut threshold ({CUT_NOISE_DB} dB)")

    mins = int(times[-1] // 60)
    ax.set_title(Path(path).name, fontsize=11)
    ax.set_xlabel("Time (minutes:seconds)")
    ax.set_ylabel("Level (dBFS)")
    ax.set_ylim(min(min(rms + peak) - 5, -80), 0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    def fmt_min(sec: float, _pos: int) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_min))
    fig.text(
        0.01, 0.01,
        f"1 sample / {times[1]-times[0]:.1f}s · green = peak below this is cut (≥{CUT_MIN_SILENCE}s gaps)",
        fontsize=8, color="#666",
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_html(path: str, times: list[float], rms: list[float], peak: list[float], out: Path) -> None:
    import json

    data = {"times": times, "rms": rms, "peak": peak, "cut_db": CUT_NOISE_DB}
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Audio dB chart</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body{{font-family:system-ui;margin:24px;background:#111;color:#eee}}
canvas{{max-width:100%;background:#1a1a1a;border-radius:8px;padding:12px}}</style></head>
<body>
<h2>{Path(path).name}</h2>
<p>Peak (red) and RMS (blue) loudness over time. Green dashed line = peak cut threshold ({CUT_NOISE_DB} dB).</p>
<canvas id="c" height="120"></canvas>
<script>
const d={json.dumps(data)};
new Chart(document.getElementById('c'),{{
 type:'line',
 data:{{
  labels:d.times.map(t=>{{const m=Math.floor(t/60),s=Math.floor(t%60);return m+':'+String(s).padStart(2,'0')}}),
  datasets:[
   {{label:'Peak dB',data:d.peak,borderColor:'#e45756',borderWidth:1,pointRadius:0}},
   {{label:'RMS dB',data:d.rms,borderColor:'#4c78a8',borderWidth:1,pointRadius:0}},
   {{label:'Cut threshold',data:d.times.map(()=>d.cut_db),borderColor:'#54a24b',borderDash:[6,4],borderWidth:2,pointRadius:0}}
  ]
 }},
 options:{{
  animation:false,
  scales:{{y:{{title:{{display:true,text:'dBFS'}},max:0}},x:{{ticks:{{maxTicksLimit:20}}}}}},
  plugins:{{legend:{{labels:{{color:'#eee'}}}}}}
 }}
}});
</script></body></html>"""
    out.write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create audio dB chart for an audio file")
    parser.add_argument("input", help="Audio file path")
    parser.add_argument("-o", "--output", help="Output PNG path (default: {stem}_db_chart.png)")
    parser.add_argument("--html", action="store_true", help="Also write interactive HTML chart")
    parser.add_argument("--window", type=float, default=1.0, help="Sample window in seconds (default: 1.0)")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Error: not found: {src}", file=sys.stderr)
        return 1

    png_out = Path(args.output) if args.output else src.with_name(f"{src.stem}_db_chart.png")
    stem_out = png_out.with_suffix("")
    ffmpeg = find_ffmpeg()

    print(f"Analysing {src.name} ({args.window}s windows)...")
    times, rms, peak = extract_levels(ffmpeg, str(src), args.window)
    report = build_report(str(src), times, rms, peak, args.window)

    save_chart(str(src), times, rms, peak, png_out)
    print(f"✓ Chart saved: {png_out}")

    json_out = stem_out.with_suffix(".json")
    save_json(report, json_out)
    print(f"✓ Data saved: {json_out}")

    csv_out = stem_out.with_suffix(".csv")
    save_csv(times, rms, peak, csv_out)
    print(f"✓ CSV saved: {csv_out}")

    if args.html:
        html_out = stem_out.with_suffix(".html")
        save_html(str(src), times, rms, peak, html_out)
        print(f"✓ HTML saved: {html_out}")

    s = report["summary"]
    print(f"  Duration: {times[-1]/60:.1f} min · samples: {len(times)}")
    print(f"  Peak above {CUT_NOISE_DB} dB: {s['peak_above_cut_threshold_pct']}% · silence gaps: {s['silence_gap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
