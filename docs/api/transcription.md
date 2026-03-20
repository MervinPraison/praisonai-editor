# Transcription API

```python
from praisonai_editor.transcribe import transcribe_audio, OpenAITranscriber

result = transcribe_audio("podcast.mp3", language="en")
```

## `transcribe_audio()`

Convenience function. Automatically chooses OpenAI or local Whisper.

```python
from praisonai_editor.transcribe import transcribe_audio

result = transcribe_audio(
    audio_path,
    use_local=False,  # True → faster-whisper
    language=None,    # None → auto-detect
)
```

## `OpenAITranscriber`

```python
from praisonai_editor.transcribe import OpenAITranscriber

transcriber = OpenAITranscriber()
result = transcriber.transcribe("audio.mp3", language="en")
```

- Automatically chunks audio > 10 minutes
- Skips chunks < 5,000 bytes (near-empty last chunk)
- 600-second timeout on the OpenAI client

## `LocalTranscriber`

```python
from praisonai_editor.transcribe import LocalTranscriber

transcriber = LocalTranscriber()
result = transcriber.transcribe("audio.mp3")
```

Requires: `pip install "praisonai-editor[local]"` (faster-whisper)

## `TranscriptResult`

```python
result.text            # full text string
result.words           # list[Word]
result.language        # "en"
result.duration        # float (seconds)

result.to_srt()        # SRT subtitle string
result.to_dict()       # dict (JSON-serializable)
```

## `Word`

```python
word.text              # "Hello"
word.start             # 0.52  (seconds)
word.end               # 1.10  (seconds)
word.confidence        # 0.99
```
