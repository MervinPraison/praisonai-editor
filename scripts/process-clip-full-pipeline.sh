#!/usr/bin/env bash
# Full sermon pipeline for one cropped clip: silence cut → normalise → transcribe → extract-text → FINAL link
# Usage: process-clip-full-pipeline.sh /path/to/clip.m4a
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLIP="${1:?usage: process-clip-full-pipeline.sh CLIP.m4a}"
STEM="$(basename "$CLIP" .m4a)"
WORKDIR="$ROOT"
LOG="$WORKDIR/${STEM}_pipeline.log"

exec > >(tee -a "$LOG") 2>&1

echo "=== Pipeline start: $STEM $(date) ==="

cd "$ROOT"
AUDIO="$CLIP"

# Step 3: silence cut
if [[ "$AUDIO" != *"_ALTERED"* ]]; then
  echo "Silence cut..."
  python3 "$ROOT/mac/cut-silence.py" "$AUDIO"
  ALTERED="${AUDIO%.*}_ALTERED${AUDIO##*.}"
  ALTERED="${AUDIO%.m4a}_ALTERED.m4a"
  [[ -f "$ALTERED" ]] || ALTERED="${AUDIO%.*}_ALTERED.wav"
  [[ -f "$ALTERED" ]] || { echo "error: silence cut output not found for $AUDIO" >&2; exit 1; }
  AUDIO="$ALTERED"
fi

# Step 4: normalise
NORM="$WORKDIR/${STEM}_norm.m4a"
echo "Normalise → $NORM"
python3 -m praisonai_editor normalize "$AUDIO" -o "$NORM"
AUDIO="$NORM"

# FINAL link
FINAL="$WORKDIR/${STEM}_FINAL.m4a"
ln -sf "$(realpath "$AUDIO")" "$FINAL"
echo "FINAL: $FINAL"

# Step 5: duration
DUR="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$FINAL")"
echo "Duration: ${DUR}s"

# Step 8: transcribe (Tamil+English auto-detect — no --language)
JSON="$WORKDIR/${STEM}_FINAL.transcript.json"
TXT="$WORKDIR/${STEM}_FINAL.transcript.txt"
if [[ -f "$JSON" && -f "$TXT" ]]; then
  echo "Transcript exists — skipping transcribe"
else
  echo "Transcribe..."
  bash -lc "cd '$ROOT' && python3 -m praisonai_editor transcribe '$FINAL' --format json -o '$JSON' 2>&1 | tee '${STEM}_FINAL_transcribe.log'"
  python3 -m praisonai_editor extract-text "$JSON"
fi

echo "=== Pipeline done: $STEM $(date) ==="
echo "Audio:  $FINAL"
echo "Text:   $TXT"
