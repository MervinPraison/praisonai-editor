# `meeting` preset

Best for: team meetings, standups, recorded calls.

## What it removes

- **Filler words**: yes, but keeps interruptions and acknowledgements intact
- **Silences**: gaps longer than **2.0 seconds** (more lenient — people think a bit more in meetings)
- **Keeps repetitions** — "Let me re-state that" is common and informative in meetings

## Usage

```bash
praisonai-editor edit standup.mp4 --preset meeting -v
```

## Python API

```python
result = edit_media("meeting.mp4", preset="meeting")
```
