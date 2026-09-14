#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPE="$ROOT/scripts/process-clip-full-pipeline.sh"

CLIPS=(
  f0cDrN8ZwVc_clip2_3727-4108.m4a
  f0cDrN8ZwVc_clip4_81525-82013.m4a
  jfSN0rEkd9k_clip2_6853-7209.m4a
  f0cDrN8ZwVc_clip3_7610-8331.m4a
  f0cDrN8ZwVc_clip1_0050-2550.m4a
  jfSN0rEkd9k_clip1_1143-4728.m4a
  xfEk4bC35-0_clip1_4215-10403.m4a
)

for clip in "${CLIPS[@]}"; do
  echo "======== BATCH: $clip ========"
  "$PIPE" "$ROOT/$clip" || echo "FAILED: $clip" >&2
done
echo "======== BATCH COMPLETE ========"
