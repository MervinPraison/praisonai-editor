# `course` preset

> Best for: online courses, lectures, tutorials, screencasts.

Best for: online courses, lectures, tutorials, screencasts.

## What it removes

- **Filler words** (strict)
- **Repetitions** (learners need clean delivery)
- **Silences** > **1.0 second** (tighter pacing for learning)

## Usage

```bash
praisonai-editor edit lecture.mp4 --preset course -v
```

## Python API

```python
result = edit_media("lecture.mp4", preset="course")
```

## Related pages

- [`meeting` preset](https://editor.praison.ai/docs/presets/meeting.md)
- [`podcast` preset](https://editor.praison.ai/docs/presets/podcast.md)
- [`speech_only` preset](https://editor.praison.ai/docs/presets/speech_only.md)
