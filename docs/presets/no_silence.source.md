# `no_silence` preset

Keep everything — just remove silent gaps between segments.

## What it keeps

- **All content**: speech, music, singing, talking over music
- Removes only **silence** blocks with no audio activity

## Usage

```bash
praisonai-editor edit recording.mp3 \
  --preset no_silence \
  --detector ensemble \
  -v
```

## When to use

- Remove dead air from a recording without changing content type
- Live recordings with gaps between songs or speakers
- Tighten up any recording without changing what's said or played

## Python API

```python
result = edit_media(
    "recording.mp3",
    preset="no_silence",
    detector="ensemble",
)
```
