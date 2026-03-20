# Agent Tools

Three built-in tools for use with PraisonAI agents. They auto-register via the `praisonaiagents.tools` entry-point.

## Available tools

| Tool name | Class | Description |
|-----------|-------|-------------|
| `audio_editor` | `AudioEditorTool` | Edit audio/video — remove fillers, silences, repetitions |
| `audio_transcribe` | `AudioTranscribeTool` | Transcribe to text with word-level timestamps |
| `mp4_to_mp3` | `MP4ToMP3Tool` | Convert MP4 (or any format) to MP3 |

## Usage with an agent

```python
from praisonaiagents import Agent
from praisonai_editor.agent_tool import (
    audio_editor_tool,
    audio_transcribe_tool,
    mp4_to_mp3_tool,
)

agent = Agent(
    instructions="You are a media processing assistant.",
    tools=[audio_editor_tool, audio_transcribe_tool, mp4_to_mp3_tool],
)

agent.start("Transcribe the interview and then clean it up for a podcast")
```

## Tool signatures

=== "AudioEditorTool"

    ```python
    audio_editor_tool.run(
        input_path="podcast.mp3",
        output_path="podcast_clean.mp3",  # optional
        preset="podcast",                  # podcast | meeting | course | clean
        use_local_whisper=False,
    )
    ```

=== "AudioTranscribeTool"

    ```python
    audio_transcribe_tool.run(
        audio_path="podcast.mp3",
        use_local_whisper=False,
        language="en",   # empty = auto-detect
    )
    # Returns: {"text": "...", "words": [...], "srt": "..."}
    ```

=== "MP4ToMP3Tool"

    ```python
    mp4_to_mp3_tool.run(
        input_path="video.mp4",
        output_path="audio.mp3",  # optional
        bitrate="192k",
    )
    # Returns: {"input_path": "...", "output_path": "..."}
    ```
