# Data Models

All models live in `praisonai_editor.models` and are plain Python dataclasses.

## ProbeResult

```python
from praisonai_editor.probe import probe_media

info = probe_media("video.mp4")
info.path               # str
info.duration           # float (seconds)
info.has_video          # bool
info.is_audio_only      # bool (property)
info.audio_codec        # "aac"
info.audio_sample_rate  # 48000
info.audio_channels     # 2
info.video_codec        # "h264"
info.width, info.height # 1920, 1080
info.fps                # 30.0
info.size_bytes         # int
info.to_dict()          # dict
```

## EditResult

```python
result.success          # bool
result.input_path       # str
result.output_path      # str
result.probe            # ProbeResult | None
result.transcript       # TranscriptResult | None
result.plan             # EditPlan | None
result.error            # str | None
result.artifacts        # dict[str, str]  {name: path}
result.to_dict()        # dict
```
