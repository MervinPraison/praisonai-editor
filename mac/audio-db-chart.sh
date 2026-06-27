#!/bin/zsh
set -euo pipefail

SCRIPT="$HOME/praisonai-audio-editor/mac/audio-db-chart.py"

for f in "$@"; do
  [[ -f "$f" ]] || continue
  python3 "$SCRIPT" "$f" --html
  html="${f%.*}_db_chart.html"
  [[ -f "$html" ]] && open "$html"
done
