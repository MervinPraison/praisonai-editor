#!/bin/zsh
set -euo pipefail

SCRIPT="$HOME/praisonai-audio-editor/mac/cut-silence.py"

for f in "$@"; do
  [[ -f "$f" ]] || continue
  python3 "$SCRIPT" "$f"
done
