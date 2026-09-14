# podcast-phone-audio — reference

## ffmpeg / ffprobe

Prefer conda (paired, smoke-tested in `praisonai_editor/probe.py`):

```bash
FF=~/miniconda3/bin/ffmpeg
FP=~/miniconda3/bin/ffprobe
```

Homebrew `/opt/homebrew/bin/ffprobe` may fail with missing `libx265` dylib.

## Noise reduction

| NR | When |
|----|------|
| 14 | Default phone memo (377–383 batch) |
| 16 | Heavier steady hiss (recording 375) |

```bash
python3 -m praisonai_editor denoise INPUT -o OUTPUT --noise-reduction 14
```

## Phone EQ filter (copy-paste)

```
highpass=f=100,
equalizer=f=250:width_type=h:width=400:g=-4,
equalizer=f=380:width_type=h:width=350:g=-2.5,
lowshelf=f=160:width_type=h:width=400:g=-1.5,
acompressor=threshold=-20dB:ratio=2.5:attack=15:release=200:makeup=2
```

## Transcription hang (root cause)

Long single Whisper API calls on 600 s Tamil chunks block 20+ min with no output. Fix: **300 s chunks** via `scripts/transcribe_recording.sh`.

## Stem naming

| Source | Stem |
|--------|------|
| `New Recording 377.m4a` | `new_recording_377` |
| `New Recording 382.m4a` | `new_recording_382` |

All work in `~/praisonai-audio-editor/`.

## Related skills

- [sermon-transcribe](../sermon-transcribe/SKILL.md) — general local audio (no phone denoise/EQ)
- [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) — YouTube download + crop
- `~/.cursor/skills/biblerevelation-sermon-articles/` — article pipeline (separate request)
