#!/bin/bash
# Transcribe {STEM}_FINAL.m4a in 5-min chunks (avoids API hang on long Tamil audio)
# Usage: transcribe_recording.sh <stem>
set -euo pipefail

STEM="${1:?stem e.g. new_recording_382}"
REPO="${REPO:-$HOME/praisonai-audio-editor}"
FF="${FF:-$HOME/miniconda3/bin/ffmpeg}"
FP="${FP:-$HOME/miniconda3/bin/ffprobe}"

AUDIO="$REPO/${STEM}_FINAL.m4a"
CHDIR="/tmp/${STEM}_chunks"
mkdir -p "$CHDIR"

DUR=$("$FP" -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$AUDIO")
N=$(python3 -c "import math; print(math.ceil(float('$DUR')/300))")

for ((i=0; i<N; i++)); do
  start=$((i*300))
  echo "[$STEM] chunk $((i+1))/$N"
  "$FF" -y -nostdin -i "$AUDIO" -ss "$start" -t 300 -c copy "$CHDIR/part_${i}.m4a" 2>/dev/null
  python3 -m praisonai_editor transcribe "$CHDIR/part_${i}.m4a" --format json -o "$CHDIR/part_${i}.json"
done

python3 << PY
import json
from pathlib import Path
stem = "$STEM"
repo = Path("$REPO")
chdir = Path("$CHDIR")
parts = sorted(chdir.glob("part_*.json"), key=lambda p: int(p.stem.split("_")[1]))
offset = 0.0
all_words = []
texts = []
for p in parts:
    d = json.loads(p.read_text())
    texts.append(d.get("text", ""))
    for w in d.get("words", []):
        all_words.append({**w, "start": w["start"]+offset, "end": w["end"]+offset})
    if d.get("words"):
        offset = all_words[-1]["end"]
    else:
        offset += d.get("duration", 300)
out = {"text": " ".join(texts), "words": all_words, "language": "ta", "duration": offset}
(repo / f"{stem}_FINAL.transcript.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(stem, "words:", len(out["text"].split()))
PY

cd "$REPO"
python3 -m praisonai_editor extract-text "${STEM}_FINAL.transcript.json"
echo "DONE transcript: ${STEM}_FINAL.transcript.txt"
