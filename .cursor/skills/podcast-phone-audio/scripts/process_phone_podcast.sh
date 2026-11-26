#!/bin/bash
# Podcast audio pipeline → {STEM}_FINAL.m4a
#
# Usage:
#   process_phone_podcast.sh <source> <stem> <start> <end> [--crop-only]
#
# Modes:
#   default     crop → denoise → phone EQ → silence cut → normalise
#   --crop-only crop → AAC only (direct/studio recording — no denoise/EQ)
#
# Example (phone memo):
#   process_phone_podcast.sh "~/Downloads/New Recording 382.m4a" new_recording_382 12:40 30:08
#
# Example (direct recording):
#   process_phone_podcast.sh "~/Downloads/150926_054013_Tr1.WAV" recording_150926_tr1 36:00 47:31 --crop-only
set -euo pipefail

SRC="${1:?source audio path}"
STEM="${2:?stem e.g. new_recording_382}"
START="${3:?crop start e.g. 12:40}"
END="${4:?crop end e.g. 30:08}"
MODE="${5:-phone}"

if [[ "$MODE" != "--crop-only" && "$MODE" != "phone" ]]; then
  echo "Unknown mode: $MODE (use --crop-only or omit for phone memo pipeline)" >&2
  exit 1
fi

REPO="${REPO:-$HOME/praisonai-audio-editor}"
FF="${FF:-$HOME/miniconda3/bin/ffmpeg}"
FP="${FP:-$HOME/miniconda3/bin/ffprobe}"

cd "$REPO"

echo "=== crop $START → $END (mode: ${MODE#--}) ==="
"$FF" -y -nostdin -i "$SRC" -ss "$START" -to "$END" -map 0:a:0 -c:a aac -b:a 192k "${STEM}_cropped.m4a"

if [[ "$MODE" == "--crop-only" ]]; then
  ln -sf "$(pwd)/${STEM}_cropped.m4a" "${STEM}_FINAL.m4a"
  "$FF" -hide_banner -nostdin -i "${STEM}_FINAL.m4a" -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume" || true
  "$FP" -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "${STEM}_FINAL.m4a" \
    | awk '{printf "FINAL duration: %d:%02d\n", int($1/60), int($1%60)}'
  echo "DONE crop-only: ${STEM}_FINAL.m4a"
  exit 0
fi

echo "=== noise QA + denoise ==="
QA_JSON="/tmp/${STEM}_noise_qa.json"
python3 -m praisonai_editor noise-qa "${STEM}_cropped.m4a" --json > "$QA_JSON"
SILENCE=$(python3 -c "import json; d=json.load(open('$QA_JSON')); print(d.get('silence_ratio',0))")
NF=$(python3 -c "import json; d=json.load(open('$QA_JSON')); print(d.get('noise_floor_db',-37))")
LEVEL=$(python3 -c "import json; d=json.load(open('$QA_JSON')); print(d.get('level','moderate'))")

# Default NR=14 — heavy mobile denoise sounds artificial; use only when MOBILE_HEAVY_DENOISE=1
if [[ "${MOBILE_HEAVY_DENOISE:-0}" == "1" ]] && python3 -c "import json; d=json.load(open('$QA_JSON')); exit(0 if float(d.get('silence_ratio',0))<0.05 and float(d.get('noise_floor_db',-60))>-42 else 1)"; then
  echo "mobile/continuous env noise (silence=${SILENCE}, floor=${NF} dB) — matched static afftdn (heavy)"
  python3 << PY
from praisonai_editor.denoise import denoise_mobile_env
denoise_mobile_env("${STEM}_cropped.m4a", "${STEM}_denoised.m4a", noise_floor_db=float("$NF"))
print("✓ Mobile env denoise done")
PY
else
  echo "standard denoise (level=${LEVEL}, NR=14) — natural speech preferred"
  python3 -m praisonai_editor denoise "${STEM}_cropped.m4a" -o "${STEM}_denoised.m4a" --noise-reduction 14 --no-track-noise
fi

echo "=== phone EQ (light makeup — avoid lifting noise bed) ==="
"$FF" -y -nostdin -i "${STEM}_denoised.m4a" \
  -af "equalizer=f=250:width_type=h:width=400:g=-4,equalizer=f=380:width_type=h:width=350:g=-2.5,lowshelf=f=160:width_type=h:width=400:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=200:makeup=1" \
  -c:a aac -b:a 192k "${STEM}_phone_opt.m4a"

echo "=== silence cut ==="
python3 mac/cut-silence.py "${STEM}_phone_opt.m4a"

echo "=== normalise (skip if already loud — normalise lifts noise floor) ==="
MEAN=$("$FF" -hide_banner -nostdin -i "${STEM}_phone_opt_ALTERED.m4a" -af volumedetect -f null - 2>&1 \
  | grep mean_volume | sed -E 's/.*mean_volume: ([-0-9.]+).*/\1/')
echo "mean_volume: ${MEAN} dB"
if python3 -c "exit(0 if float('${MEAN}') < -22 else 1)"; then
  python3 -m praisonai_editor normalize "${STEM}_phone_opt_ALTERED.m4a" -o "${STEM}_norm.m4a"
  ln -sf "$(pwd)/${STEM}_norm.m4a" "${STEM}_FINAL.m4a"
else
  echo "skip normalise (mean >= -22 dB) — use denoised output to avoid lifting room noise"
  ln -sf "$(pwd)/${STEM}_phone_opt_ALTERED.m4a" "${STEM}_FINAL.m4a"
fi
"$FF" -hide_banner -nostdin -i "${STEM}_FINAL.m4a" -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume" || true
"$FP" -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "${STEM}_FINAL.m4a" \
  | awk '{printf "FINAL duration: %d:%02d\n", int($1/60), int($1%60)}'

echo "DONE audio: ${STEM}_FINAL.m4a"
