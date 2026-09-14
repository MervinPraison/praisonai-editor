#!/usr/bin/env bash
# Validate cropped clips: file exists, ffprobe readable, duration within tolerance.
# Usage: validate-youtube-clips.sh START END FILE [START END FILE ...]
set -euo pipefail

die() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "OK  $*"; }

timestamp_to_seconds() {
  python3 - "$1" <<'PY'
import sys
parts = [int(p) for p in sys.argv[1].split(":")]
if len(parts) == 2:
    h, m, s = 0, parts[0], parts[1]
else:
    h, m, s = parts
print(h * 3600 + m * 60 + s)
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

validate_one() {
  local start="$1" end="$2" file="$3"
  [[ -f "$file" ]] || die "missing file: $file"

  local expected actual
  expected="$(($(timestamp_to_seconds "$end") - $(timestamp_to_seconds "$start")))"
  actual="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")"

  python3 - "$expected" "$actual" "$file" <<'PY'
import sys
expected = float(sys.argv[1])
actual = float(sys.argv[2])
file = sys.argv[3]
delta = abs(actual - expected)
if delta > 2.0:
    raise SystemExit(f"{file}: expected ~{expected:.1f}s, got {actual:.1f}s (delta {delta:.1f}s)")
print(f"expected={expected:.1f}s actual={actual:.1f}s delta={delta:.2f}s")
PY

  pass "$(basename "$file") $(format_duration "$actual") (expected $(format_duration "$expected"))"
}

main() {
  (( $# % 3 == 0 )) || die "usage: validate-youtube-clips.sh START END FILE [...]"
  command -v ffprobe >/dev/null || die "ffprobe not found"

  local failures=0
  while (( $# > 0 )); do
    if ! validate_one "$1" "$2" "$3"; then
      failures=$((failures + 1))
    fi
    shift 3
  done

  (( failures == 0 )) || die "${failures} clip(s) failed validation"
  echo "All clips passed."
}

main "$@"
