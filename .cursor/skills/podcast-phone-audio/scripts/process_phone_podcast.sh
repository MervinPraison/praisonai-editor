#!/bin/bash
# Crop + denoise + phone EQ + silence cut + normalise → {STEM}_FINAL.m4a
# Usage: process_phone_podcast.sh <source.m4a> <stem> <start> <end>
# Example: process_phone_podcast.sh "/Users/praison/Downloads/New Recording 382.m4a" new_recording_382 12:40 30:08
set -euo pipefail

SRC="${1:?source audio path}"
STEM="${2:?stem e.g. new_recording_382}"
START="${3:?crop start e.g. 12:40}"
END="${4:?crop end e.g. 30:08}"

REPO="${REPO:-$HOME/praisonai-audio-editor}"
FF="${FF:-$HOME/miniconda3/bin/ffmpeg}"
FP="${FP:-$HOME/miniconda3/bin/ffprobe}"

cd "$REPO"

echo "=== crop $START → $END ==="
"$FF" -y -nostdin -i "$SRC" -ss "$START" -to "$END" -map 0:a:0 -c:a aac -b:a 192k "${STEM}_cropped.m4a"

echo "=== denoise (NR=14) ==="
python3 -m praisonai_editor denoise "${STEM}_cropped.m4a" -o "${STEM}_denoised.m4a" --noise-reduction 14

echo "=== phone EQ ==="
"$FF" -y -nostdin -i "${STEM}_denoised.m4a" \
  -af "highpass=f=100,equalizer=f=250:width_type=h:width=400:g=-4,equalizer=f=380:width_type=h:width=350:g=-2.5,lowshelf=f=160:width_type=h:width=400:g=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=200:makeup=2" \
  -c:a aac -b:a 192k "${STEM}_phone_opt.m4a"

echo "=== silence cut ==="
python3 mac/cut-silence.py "${STEM}_phone_opt.m4a"

echo "=== normalise ==="
"$FF" -hide_banner -nostdin -i "${STEM}_phone_opt_ALTERED.m4a" -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume" || true
python3 -m praisonai_editor normalize "${STEM}_phone_opt_ALTERED.m4a" -o "${STEM}_norm.m4a"
"$FF" -hide_banner -nostdin -i "${STEM}_norm.m4a" -af volumedetect -f null - 2>&1 | grep -E "mean_volume|max_volume" || true

ln -sf "$(pwd)/${STEM}_norm.m4a" "${STEM}_FINAL.m4a"
"$FP" -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "${STEM}_FINAL.m4a" \
  | awk '{printf "FINAL duration: %d:%02d\n", int($1/60), int($1%60)}'

echo "DONE audio: ${STEM}_FINAL.m4a"
