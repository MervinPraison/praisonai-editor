#!/usr/bin/env bash
# Step 2b — normalise quiet YouTube / sermon audio before transcribe.
# Usage: optimize-audio-volume.sh /path/to/audio.m4a [--in-place] [--force]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 -m praisonai_editor normalize "$@"
