# `course` preset

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
