---
name: youtube-clip-crop
description: Downloads YouTube audio and crops one or more timestamp ranges to m4a files only — no transcription, normalisation, or articles. Use when the user gives a YouTube URL with start/end times, asks to crop clips, extract audio segments, or says they only want cropped audio.
---

# YouTube clip crop (audio only)

Run every step on the user's Mac. **Do not stop at instructions.**

**Scope:** download + crop + deliver m4a files. **Do not** transcribe, extract text, normalise volume, merge clips, or create articles unless the user explicitly asks.

For the full pipeline (normalise → transcribe → pack), use [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md).

## Prerequisites

| Tool | Check |
|------|--------|
| `ffmpeg` / `ffprobe` | `ffmpeg -version` |
| `yt-dlp` | **conda** `python3 -m yt_dlp` (homebrew `yt-dlp` often returns **403**) |
| Output dir | `~/praisonai-audio-editor` |

## Parse user input

1. **YouTube URL** → extract `{VIDEO_ID}` from `watch?v=` or `youtu.be/`
2. **One or more ranges** — each `START` → `END` pair (accept `M:SS`, `H:MM:SS`, `1:8:53` = `1:08:53`)
3. **`-to` is absolute** on the full video timeline (not relative duration)
4. **"till the end"** — `-ss START` only (omit `-to`)
5. **Multiple URLs** — repeat download + crop per video

## Paths and naming

| Artifact | Path |
|----------|------|
| Full download | `~/Downloads/{VIDEO_ID}_full.m4a` |
| Cropped clip | `~/praisonai-audio-editor/{VIDEO_ID}_clip{N}_{START}-{END}.m4a` |

**Compact timestamp in filename** — strip colons, zero-pad minutes/seconds to 2 digits:
- `0:50` → `0050`, `25:50` → `2550` → `0050-2550`
- `1:08:53` → `6853`, `11:43` → `1143`
- Clip index `{N}`: 1, 2, 3… per video, in user order

Example: `f0cDrN8ZwVc_clip1_0050-2550.m4a`

## Checklist

```
- [ ] Skip download if ~/Downloads/{VIDEO_ID}_full.m4a exists
- [ ] Download (conda yt-dlp) if missing
- [ ] Crop each range (ffmpeg stream copy)
- [ ] ffprobe duration check per clip
- [ ] Reply with reference table
```

## Step 1 — Download

Skip if `~/Downloads/{VIDEO_ID}_full.m4a` already exists.

```bash
zsh -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate test && \
  python3 -m yt_dlp -f 'ba/b' --no-playlist --concurrent-fragments 8 --no-part \
  -x --audio-format m4a \
  -o '/Users/praison/Downloads/{VIDEO_ID}_full.%(ext)s' 'URL'"
```

403 → `pip install -U yt-dlp` + `--extractor-args 'youtube:player_client=android,web'`.  
**No `--print` on download** — it can skip writing the file.

## Step 2 — Crop

One ffmpeg call per range:

```bash
FULL="$HOME/Downloads/{VIDEO_ID}_full.m4a"
OUT="$HOME/praisonai-audio-editor"

ffmpeg -y -nostdin -i "$FULL" -ss START -to END -map 0:a:0 -c copy \
  "$OUT/{VIDEO_ID}_clip{N}_{START}-{END}.m4a"
```

End of video: omit `-to`.

## Step 3 — Verify

```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 CLIP.m4a
```

Expected duration ≈ `END − START`. Report `H:MM:SS` in the reference table.

## Step 4 — Reference table (always)

End every run with this table for all clips produced:

| File name | URL | From | To | Duration |
|-----------|-----|------|-----|----------|
| `{VIDEO_ID}_clip1_….m4a` | full YouTube URL | start | end | H:MM:SS |

Include full paths in a follow-up line or separate column if helpful: `~/praisonai-audio-editor/…`

## Do not use

- Homebrew `yt-dlp` alone (403)
- `--download-sections` on long clips
- Transcription / `praisonai_editor transcribe` (unless user asks)
- Volume normalisation (unless user asks)
- Merging clips (unless user asks)

## Optional follow-ups (only if requested)

| Request | Action |
|---------|--------|
| Merge clips | ffmpeg concat → single m4a |
| Normalise volume | `python3 -m praisonai_editor normalize` or [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) |
| Transcribe | [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) or [sermon-transcribe](../sermon-transcribe/SKILL.md) |

## Examples

See [examples.md](examples.md).
