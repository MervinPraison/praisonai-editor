# `podcast` preset

Best for: podcast recordings, interviews, solo recordings.

## What it removes

- **Filler words**: um, uh, er, ah, like, you know, basically, actually, literally…
- **Repetitions**: same word repeated within 3 words
- **Silences**: gaps longer than **1.5 seconds**

## Usage

```bash
praisonai-editor edit episode.mp3 --preset podcast -v
```

## Typical result

| Before | After | Saved |
|--------|-------|-------|
| 45 min | ~38 min | ~15% |

## Python API

```python
from praisonai_editor.pipeline import edit_media

result = edit_media("episode.mp3", preset="podcast")
print(f"Saved {result.plan.removed_duration:.0f}s")
```
