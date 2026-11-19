# `no_silence` preset

> Keep everything — just remove silent gaps between segments.

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

## Related pages

- [`speech_only` preset](https://editor.praison.ai/docs/presets/speech_only.md)
- [`songs_only` preset](https://editor.praison.ai/docs/presets/songs_only.md)
- [`convert` — Convert between formats](https://editor.praison.ai/docs/commands/convert.md)
