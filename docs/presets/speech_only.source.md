# `speech_only` preset

Extract only speech portions — removes music, singing, and silence.

## What it keeps

- **Speech** segments (clear talking without music)
- **Talking over music** (speech overlaid on music)
- **Speech over music** (general speech+music mix)

## Usage

```bash
praisonai-editor edit radio_show.mp3 \
  --preset speech_only \
  --detector ensemble \
  -v
```

## When to use

- Radio shows where you want interview clips without music beds
- Live stream recordings — keep only the talking parts
- Music shows — extract the host commentary only

!!! note
    Requires `--detector` (a content detector). Use `ensemble` for best results.

## Python API

```python
from praisonai_editor.pipeline import edit_media

result = edit_media(
    "radio.mp3",
    preset="speech_only",
    detector="ensemble",
    verbose=True,
)
```
