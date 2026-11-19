# `edit_media()` — Main API

```python
from praisonai_editor.pipeline import edit_media

result = edit_media(input_path, output_path=None, **kwargs)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | required | Path to audio or video file |
| `output_path` | `str` | auto | Output path (default: `{stem}_edited.{ext}`) |
| `preset` | `str` | `"podcast"` | Edit preset |
| `detector` | `str` | `"auto"` | Content detector |
| `demix` | `bool` | `False` | Enable Demucs stem separation |
| `primary_zone_only` | `bool` | `False` | Crop to primary singing zone |
| `remove_fillers` | `bool` | `True` | Remove filler words |
| `remove_repetitions` | `bool` | `True` | Remove repeated words |
| `remove_silence` | `bool` | `True` | Remove long silences |
| `min_silence` | `float` | `1.5` | Minimum silence duration (seconds) |
| `use_local_whisper` | `bool` | `False` | Use faster-whisper (offline) |
| `language` | `str` | None | Language code (`en`, `ta`, `es`, …) |
| `copy_codec` | `bool` | `True` | Copy codec (faster vs re-encode) |
| `verbose` | `bool` | `False` | Print progress |
| `save_artifacts` | `bool` | `True` | Save transcript/plan/blocks to disk |

## Returns: `EditResult`

```python
result.success         # bool
result.output_path     # str
result.input_path      # str
result.probe           # ProbeResult
result.transcript      # TranscriptResult
result.plan            # EditPlan
result.error           # str | None
result.artifacts       # dict[str, str]
```

## Examples

=== "Simple"

    ```python
    from praisonai_editor.pipeline import edit_media

    result = edit_media("podcast.mp3")
    ```

=== "Music extraction"

    ```python
    result = edit_media(
        "concert.mp3",
        preset="songs_only",
        detector="ensemble",
        demix=True,
        primary_zone_only=True,
        verbose=True,
    )
    ```

=== "Custom silence threshold"

    ```python
    result = edit_media(
        "recording.mp3",
        preset="podcast",
        min_silence=2.0,
        remove_fillers=False,
    )
    ```
