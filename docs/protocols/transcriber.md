# Custom Transcriber

Replace OpenAI Whisper with any speech-to-text engine.

## Protocol

```python
from praisonai_editor.protocols import Transcriber
from praisonai_editor.models import TranscriptResult, Word

class MyTranscriber:
    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
    ) -> TranscriptResult:
        ...
```

## Example: AssemblyAI

```python
import assemblyai as aai
from praisonai_editor.models import TranscriptResult, Word

class AssemblyAITranscriber:
    def transcribe(self, audio_path, *, language=None):
        transcriber = aai.Transcriber()
        t = transcriber.transcribe(audio_path)
        words = [
            Word(text=w.text, start=w.start/1000, end=w.end/1000)
            for w in t.words
        ]
        return TranscriptResult(text=t.text, words=words, language=language or "en")

# Use it in the pipeline manually
from praisonai_editor.probe import FFmpegProber
from praisonai_editor.plan import create_edit_plan
from praisonai_editor.render import FFmpegAudioRenderer

prober = FFmpegProber()
probe = prober.probe("podcast.mp3")

transcriber = AssemblyAITranscriber()
transcript = transcriber.transcribe("podcast.mp3")

plan = create_edit_plan(transcript, probe.duration, preset="podcast")

renderer = FFmpegAudioRenderer()
renderer.render("podcast.mp3", "podcast_edited.mp3", plan)
```
