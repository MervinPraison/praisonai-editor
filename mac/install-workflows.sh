#!/bin/zsh
set -euo pipefail

REPO="$HOME/praisonai-audio-editor/mac"
SERVICES="$HOME/Library/Services"

chmod +x "$REPO/autoedit-audio.sh" "$REPO/audio-edit-ai.sh" "$REPO/audio-db-chart.sh" "$REPO/audio-db-chart.py"

cp -R "$REPO/autoedit-audio.workflow" "$SERVICES/"
cp -R "$REPO/Audio Edit (AI).workflow" "$SERVICES/"
cp -R "$REPO/audio-db-chart.workflow" "$SERVICES/"

# Replace legacy Audio Edit if present
if [[ -d "$SERVICES/Audio Edit.workflow" ]]; then
  rm -rf "$SERVICES/Audio Edit.workflow"
fi

enable_quick_action() {
  python3 <<'PY'
import plistlib
from pathlib import Path

plist_path = Path.home() / "Library/Preferences/pbs.plist"
with plist_path.open("rb") as f:
    data = plistlib.load(f)

services = data.setdefault("NSServicesStatus", {})
entry = {
    "presentation_modes": {
        "ContextMenu": 1,
        "FinderPreview": 1,
        "ServicesMenu": 1,
        "TouchBar": 1,
    }
}
for name in ("autoedit-audio", "Audio Edit (AI)", "Audio dB Chart"):
    services[f"(null) - {name} - runWorkflowAsService"] = entry

with plist_path.open("wb") as f:
    plistlib.dump(data, f)
PY
}

enable_quick_action

/System/Library/CoreServices/pbs -flush
killall Finder 2>/dev/null || true

echo "Installed:"
echo "  $SERVICES/autoedit-audio.workflow"
echo "  $SERVICES/Audio Edit (AI).workflow"
echo "  $SERVICES/audio-db-chart.workflow"
echo ""
echo "Quick Actions enabled for Finder context menu."
