# `clean` preset

> Maximum cleanup. Tightest silence removal.

Maximum cleanup. Tightest silence removal.

## What it removes

- **Filler words**
- **Repetitions**
- **Silences** > **0.8 seconds** — very tight, produces a fast-paced output

## Usage

```bash
praisonai-editor edit raw.mp3 --preset clean -v
```

!!! warning "Aggressive"
    0.8s silence threshold is aggressive. Rapid-fire speech may feel unnatural.
    Use `course` (1.0s) or `podcast` (1.5s) if results feel rushed.

## Related pages

- [`speech_only` preset](https://editor.praison.ai/docs/presets/speech_only.md)
- [`no_silence` preset](https://editor.praison.ai/docs/presets/no_silence.md)
- [`songs_only` preset](https://editor.praison.ai/docs/presets/songs_only.md)
