# Installation

## Requirements

- Python **3.10+**
- **FFmpeg** installed on your system
- **OpenAI API key** (for cloud transcription, or use `--local`)

---

## Install FFmpeg

=== "macOS"
    ```bash
    brew install ffmpeg
    ```

=== "Ubuntu / Debian"
    ```bash
    sudo apt install ffmpeg
    ```

=== "Windows"
    Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

---

## Install praisonai-editor

### Minimal (OpenAI Whisper)
```bash
pip install praisonai-editor
```

### With local Whisper (offline)
```bash
pip install "praisonai-editor[local]"
```

### With Demucs stem separation
```bash
pip install "praisonai-editor[demix]"
```

### With INA speech detector
```bash
pip install "praisonai-editor[detect]"
```

### Everything
```bash
pip install "praisonai-editor[all]"
```

---

## Extras at a glance

| Extra | What it adds | When to use |
|-------|-------------|-------------|
| *(default)* | OpenAI Whisper cloud | Always use first |
| `[local]` | faster-whisper (offline) | No internet / privacy |
| `[demix]` | Demucs stem separation | Music files, `--demix` |
| `[detect]` | inaSpeechSegmenter | `--detector ina` |
| `[all]` | All of the above | Power users |

---

## Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

!!! tip "No API key needed"
    Use `--local` flag to transcribe completely offline with faster-whisper:
    ```bash
    praisonai-editor edit file.mp3 --local
    ```

---

## Verify installation

```bash
praisonai-editor --help
praisonai-editor probe myfile.mp3
```
