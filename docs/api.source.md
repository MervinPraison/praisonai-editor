# Python API Overview

Use praisonai-editor directly from Python — no CLI needed.

## Core entry point

```python
from praisonai_editor.pipeline import edit_media

result = edit_media("podcast.mp3", preset="podcast", verbose=True)
print(result.output_path)    # "podcast_edited.mp3"
print(result.success)        # True
```

## Pipeline flow

```mermaid
flowchart LR
    A["edit_media()"] --> B[auto-detect\naudio vs video]
    B -->|audio| C["edit_audio()"]
    B -->|video| D["edit_video()"]
    C --> E[probe → transcribe\n→ plan → render]
    D --> E
```

## Module layout

| Module | Key exports |
|--------|------------|
| `pipeline` | `edit_media`, `edit_audio`, `edit_video` |
| `probe` | `probe_media`, `FFmpegProber` |
| `transcribe` | `transcribe_audio`, `OpenAITranscriber`, `LocalTranscriber` |
| `convert` | `convert_media` |
| `plan` | `create_edit_plan`, `HeuristicEditor` |
| `detect` | `create_content_plan` |
| `render` | `FFmpegAudioRenderer`, `FFmpegVideoRenderer` |
| `_demix` | `isolate_vocals`, `has_demucs` |
| `models` | `EditPlan`, `EditResult`, `TranscriptResult`, `ProbeResult` |
| `protocols` | `Transcriber`, `Editor`, `Renderer`, `Prober`, `Converter` |
