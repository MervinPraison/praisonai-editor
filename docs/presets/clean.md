# `clean` preset

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
