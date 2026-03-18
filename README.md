# PraisonAI Editor

AI-powered audio & video editor for [PraisonAI](https://docs.praison.ai). Protocol-driven, modular, and ready for agent integration.

## Install

```bash
pip install praisonai-editor
```

With local Whisper support:
```bash
pip install "praisonai-editor[local]"
```

With PraisonAI Agent integration:
```bash
pip install "praisonai-editor[agent]"
```

## Quick Start

### CLI

```bash
# Edit audio (transcribe → detect fillers/silence → render)
praisonai-editor edit podcast.mp3 --preset podcast --verbose

# Edit video
praisonai-editor edit interview.mp4 --output edited.mp4

# AI-guided editing with prompt
praisonai-editor edit podcast.mp3 --prompt "Remove intro and off-topic weather discussion"

# Transcribe
praisonai-editor transcribe audio.mp3 --format srt

# Convert MP4 → MP3
praisonai-editor convert video.mp4 --format mp3

# Probe file metadata
praisonai-editor probe media.mp4
```

### Python API

```python
from praisonai_editor import edit_media, transcribe_audio, convert_media

# Edit any media
result = edit_media("podcast.mp3", preset="podcast")

# Transcribe
transcript = transcribe_audio("audio.mp3")
print(transcript.to_srt())

# Convert
convert_media("video.mp4", "audio.mp3")
```

### PraisonAI Agent

```python
from praisonaiagents import Agent
from praisonai_editor.agent_tool import audio_editor_tool

agent = Agent(
    instructions="You are an audio editor.",
    tools=[audio_editor_tool],
)
```

## Presets

| Preset | Fillers | Repetitions | Silence (min) |
|--------|---------|-------------|---------------|
| podcast | ✅ Remove | ✅ Remove | 1.5s |
| meeting | ✅ Remove | Keep | 2.0s |
| course | ✅ Remove | ✅ Remove | 1.0s |
| clean | ✅ Remove | ✅ Remove | 0.8s |

## Requirements

- Python ≥ 3.10
- FFmpeg (for audio/video processing)
- OpenAI API key (for transcription) or `[local]` extra for offline Whisper
