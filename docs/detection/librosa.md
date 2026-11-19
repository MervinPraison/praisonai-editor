# Librosa spectral detector (`--detector librosa`)

> Uses spectral features (RMS energy, spectral centroid, zero-crossing rate) from librosa to classify segments.

Uses spectral features (RMS energy, spectral centroid, zero-crossing rate) from librosa to classify segments.

## Usage

```bash
praisonai-editor edit file.mp3 --preset songs_only --detector librosa
```

## How it works

```mermaid
flowchart LR
    A[Audio frame] --> B[RMS energy dB]
    A --> C[Spectral centroid]
    A --> D[Zero-crossing rate]
    B --> E[Classify frame]
    C --> E
    D --> E
    E --> F{speech, music,\nsilence, singing?}
```

## Strengths

- No extra install (librosa is a base dependency after install)
- Fast — runs on CPU without ML inference
- Works fine for clean recordings

## Weaknesses

- Less accurate than ensemble or INA on noisy or complex audio
- May confuse speech over music as pure music

## Related pages

- [FFmpeg heuristic detector (`--detector ffmpeg`)](https://editor.praison.ai/docs/detection/ffmpeg.md)
- [INA Speech Segmenter (`--detector ina`)](https://editor.praison.ai/docs/detection/ina.md)
- [Ensemble detector (`--detector ensemble`)](https://editor.praison.ai/docs/detection/ensemble.md)
