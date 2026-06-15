---
name: youtube-clip-transcribe
description: Download YouTube audio, crop by start/end timestamps with ffmpeg, transcribe with praisonai-audio-editor, extract plain text to .txt. Use whenever the user provides a YouTube URL with start and finish times, a time range, or clip boundaries (one or many videos).
---

# YouTube URL + start/end times → crop → transcribe

Run every step on the user's Mac. **Do not stop at instructions.**

For **multiple URLs**, repeat the pipeline per video — do not mix transcripts.

## Prerequisites

| Tool | Check |
|------|--------|
| `yt-dlp` | `yt-dlp --version` |
| `ffmpeg` / `ffprobe` | `ffmpeg -version` |
| `OPENAI_API_KEY` | `bash -lc 'test -n "$OPENAI_API_KEY" && echo ok'` |
| This repo | `~/praisonai-audio-editor` |

**Shell:** use `bash -lc '…'` so `~/.bashrc` loads `OPENAI_API_KEY`.

## Parse user input

1. **YouTube URL** — extract `{VIDEO_ID}` from `watch?v=`, `youtu.be/`, `live/`
2. **Start** (inclusive) and **End** / **Finish** — ffmpeg `-to` is **absolute** on the full timeline
3. **Output dir** (optional) — default repo root `~/praisonai-audio-editor/`

Timestamp forms: `41:58`, `1:36:41`, `1:01:05`. Normalise to `HH:MM:SS[.fff]`.

## Output files (one stem per clip)

| Step | Path |
|------|------|
| Full download | `~/Downloads/{VIDEO_ID}_full.m4a` |
| Cropped audio | `~/praisonai-audio-editor/{VIDEO_ID}_{start}_to_{end}.m4a` |
| Transcript JSON | `…/{VIDEO_ID}_{start}_to_{end}.transcript.json` |
| Plain text | `…/{VIDEO_ID}_{start}_to_{end}.transcript.txt` |
| Log | `…/{VIDEO_ID}_{start}_to_{end}_transcribe.log` |

Example: `VCasP2Z2wQM_41m58_to_1h36m41.m4a`

## Checklist

```
Per YouTube URL:
- [ ] Download full audio (NOT --download-sections for long clips)
- [ ] Crop with ffmpeg (-c copy)
- [ ] Verify duration (ffprobe ≈ end − start)
- [ ] Transcribe → JSON
- [ ] Extract plain text → .txt
- [ ] Sanity-check: JSON duration ≈ ffprobe; preview open/close text

Final report:
- [ ] YouTube URL, crop window, all output paths
- [ ] Duration, word count, opening/closing sentences
```

## Step 1 — Download

```bash
bash -lc 'yt-dlp -f "ba/b" --no-playlist --concurrent-fragments 8 --no-part \
  -x --audio-format m4a \
  -o "/Users/praison/Downloads/{VIDEO_ID}_full.%(ext)s" \
  "YOUTUBE_URL"'
```

Add `--cookies-from-browser chrome` if needed.

## Step 2 — Crop

```bash
ffmpeg -y -nostdin -i "/Users/praison/Downloads/{VIDEO_ID}_full.m4a" \
  -ss "START" -to "END" \
  -map 0:a:0 -c copy \
  "/Users/praison/praisonai-audio-editor/{VIDEO_ID}_{start}_to_{end}.m4a"
```

Verify: `ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 CROPPED.m4a`

## Step 3 — Transcribe

```bash
bash -lc 'cd ~/praisonai-audio-editor && python3 -m praisonai_editor transcribe \
  "CROPPED.m4a" --format json --language en \
  -o "CROPPED.transcript.json" 2>&1 | tee "CROPPED_transcribe.log"'
```

## Step 4 — Extract text

```bash
cd ~/praisonai-audio-editor
python3 -m praisonai_editor extract-text "CROPPED.transcript.json"
```

Or in one step from audio: `transcribe … --format txt -o out.txt` (runs ASR again).

## Optional (only if user asks)

| Request | Action |
|---------|--------|
| WordPress / biblerevelation article | Write `.agent/biblerevelation-{slug}.html` from transcript; `apply_highlights.py --yaml … --html …` |
| Phrase cut instead of timestamps | `python3 -m praisonai_editor trim …` — see `docs/commands/transcribe.md` |

**Do not** create per-sermon Python recipe files. Articles = `.html` drafts from transcript.

## Do not use

| Tool | Why |
|------|-----|
| `yt-dlp --download-sections` on 30–60 min clips | Real-time re-encode |
| `transcribe --force-transcribe` | Flag does not exist on `transcribe` |

## More detail

Extended reference, examples, and WordPress publish flow: `~/.cursor/skills/youtube-clip-transcribe/` (`reference.md`, `examples.md`, `biblerevelation-publish.md`)
