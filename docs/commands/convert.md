# `convert` — Convert between formats

Convert between audio and video formats using FFmpeg.

## Usage

```bash
praisonai-editor convert INPUT [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `INPUT` | | | Source media file |
| `--output` | `-o` | `{stem}.{format}` | Output file path |
| `--format` | `-f` | `mp3` | Target format: `mp3`, `wav`, `m4a` |
| `--bitrate` | `-b` | `192k` | Audio bitrate |

## Examples

=== "MP4 → MP3"

    ```bash
    praisonai-editor convert video.mp4
    # → video.mp3 (192k)
    ```

=== "High quality"

    ```bash
    praisonai-editor convert video.mp4 --bitrate 320k
    ```

=== "WAV output"

    ```bash
    praisonai-editor convert podcast.mp3 --format wav
    ```

=== "Custom output path"

    ```bash
    praisonai-editor convert video.mp4 --output ~/Desktop/audio.mp3
    ```

## Bitrate guide

| Bitrate | Use case |
|---------|----------|
| `64k` | Voice / speech (small file) |
| `128k` | Standard quality |
| `192k` | Good quality *(default)* |
| `320k` | High quality / music |

!!! tip "Python API"
    ```python
    from praisonai_editor.convert import convert_media

    output = convert_media("video.mp4", "audio.mp3", bitrate="192k")
    print(output)  # "audio.mp3"
    ```
