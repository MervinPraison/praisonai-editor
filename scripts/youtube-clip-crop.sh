#!/usr/bin/env bash
# Download YouTube audio and crop timestamp ranges to m4a.
# Usage: youtube-clip-crop.sh URL START END [START END ...]
# Example: youtube-clip-crop.sh 'https://youtube.com/watch?v=ID' 0:50 25:50 37:27 41:08
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOWNLOADS="${HOME}/Downloads"
OUT="${ROOT}"
FFMPEG="${FFMPEG:-$(command -v ffmpeg)}"
FFPROBE="${FFPROBE:-$(command -v ffprobe)}"

die() { echo "error: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$1 not found"
}

timestamp_to_seconds() {
  python3 - "$1" <<'PY'
import sys
parts = sys.argv[1].strip().split(":")
parts = [int(p) for p in parts]
if len(parts) == 2:
    m, s = parts
    h = 0
elif len(parts) == 3:
    h, m, s = parts
else:
    raise SystemExit(f"bad timestamp: {sys.argv[1]!r}")
print(h * 3600 + m * 60 + s)
PY
}

compact_timestamp() {
  python3 - "$1" <<'PY'
import sys
parts = sys.argv[1].strip().split(":")
parts = [int(p) for p in parts]
if len(parts) == 2:
    h, m, s = 0, parts[0], parts[1]
else:
    h, m, s = parts
total_minutes = h * 60 + m
print(f"{total_minutes:02d}{s:02d}" if total_minutes < 100 else f"{total_minutes}{s:02d}")
PY
}

format_duration() {
  python3 - "$1" <<'PY'
import sys
sec = float(sys.argv[1])
h = int(sec // 3600)
m = int((sec % 3600) // 60)
s = int(sec % 60)
print(f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}")
PY
}

extract_video_id() {
  python3 - "$1" <<'PY'
import re, sys
url = sys.argv[1]
m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
if not m:
    raise SystemExit(f"could not parse video id from {url!r}")
print(m.group(1))
PY
}

ensure_full_m4a() {
  local video_id="$1" url="$2"
  local m4a="${DOWNLOADS}/${video_id}_full.m4a"
  local webm="${DOWNLOADS}/${video_id}_full.webm"

  if [[ -f "$m4a" ]]; then
    echo "$m4a"
    return
  fi

  if [[ ! -f "$webm" ]]; then
    echo "Downloading ${video_id}..." >&2
    zsh -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate test && \
      python3 -m yt_dlp -f 'ba/b' --no-playlist --concurrent-fragments 8 --no-part \
      --ffmpeg-location '${FFMPEG}' \
      -x --audio-format m4a \
      -o '${DOWNLOADS}/${video_id}_full.%(ext)s' '${url}'" >&2
  fi

  if [[ -f "$m4a" ]]; then
    echo "$m4a"
    return
  fi

  if [[ -f "$webm" ]]; then
    echo "Converting ${webm} to m4a..." >&2
    "$FFMPEG" -loglevel error -y -nostdin -i "$webm" -c:a aac -b:a 192k "$m4a"
    echo "$m4a"
    return
  fi

  die "download failed for ${video_id}"
}

crop_clip() {
  local full="$1" video_id="$2" index="$3" start="$4" end="$5" start_sec="$6" end_sec="$7"
  local start_compact end_compact out expected actual

  start_compact="$(compact_timestamp "$start")"
  end_compact="$(compact_timestamp "$end")"
  out="${OUT}/${video_id}_clip${index}_${start_compact}-${end_compact}.m4a"

  "$FFMPEG" -loglevel error -y -nostdin -i "$full" -ss "$start" -to "$end" -map 0:a:0 -c copy "$out"

  expected="$((end_sec - start_sec))"
  actual="$("$FFPROBE" -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$out")"
  python3 - "$expected" "$actual" <<'PY'
import sys
expected = float(sys.argv[1])
actual = float(sys.argv[2])
if abs(actual - expected) > 2.0:
    raise SystemExit(f"duration mismatch: expected ~{expected:.1f}s, got {actual:.1f}s")
PY

  printf '%s\n' "$out"
}

print_table_row() {
  local file="$1" url="$2" start="$3" end="$4" duration="$5"
  printf '| `%s` | %s | %s | %s | %s |\n' "$(basename "$file")" "$url" "$start" "$end" "$duration"
}

main() {
  [[ $# -ge 3 ]] || die "usage: youtube-clip-crop.sh URL START END [START END ...]"
  require_cmd ffmpeg
  require_cmd ffprobe
  require_cmd python3

  local url="$1"
  shift
  (( $# % 2 == 0 )) || die "timestamp ranges must be START END pairs"

  local video_id full index=0
  video_id="$(extract_video_id "$url")"
  full="$(ensure_full_m4a "$video_id" "$url")"

  echo ""
  echo "| File name | URL | From | To | Duration |"
  echo "|-----------|-----|------|-----|----------|"

  while (( $# > 0 )); do
    local start="$1" end="$2"
    shift 2
    index=$((index + 1))
    start_seconds="$(timestamp_to_seconds "$start" | tr -d '[:space:]')"
    end_seconds="$(timestamp_to_seconds "$end" | tr -d '[:space:]')"
    out="$(crop_clip "$full" "$video_id" "$index" "$start" "$end" "$start_seconds" "$end_seconds")"
    dur="$(format_duration "$((end_seconds - start_seconds))")"
    print_table_row "$out" "$url" "$start" "$end" "$dur"
    echo "Saved: ${out}" >&2
  done
}

main "$@"
