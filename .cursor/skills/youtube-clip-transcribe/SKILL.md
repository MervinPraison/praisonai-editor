---
name: youtube-clip-transcribe
description: Downloads YouTube sermon audio, crops by timestamps, optimises volume (quiet or hot), merges multi-part clips, transcribes with praisonai-audio-editor, maps to YAML decks, and packs deliverables. Use when the user gives a YouTube URL with start/end times, full video, two-part merge, sermon transcript, or BIC sermon deck pack.
---

# YouTube clip → normalise → transcribe → pack

Run every step on the user's Mac. **Do not stop at instructions.**

Repeat the pipeline per clip. **Never merge transcripts** unless the user explicitly asks to join two videos into one audio file first.

## Prerequisites

| Tool | Check |
|------|--------|
| `ffmpeg` / `ffprobe` | `ffmpeg -version` |
| `OPENAI_API_KEY` | `bash -lc 'test -n "$OPENAI_API_KEY" && echo ok'` |
| This repo | `~/praisonai-audio-editor` |
| `yt-dlp` | **conda** `python3 -m yt_dlp` (homebrew `yt-dlp` often returns **403**) |
| YAML decks | `/Users/praison/praisonaippt/examples/*.yaml` |

**Shell:** use `bash -lc '…'` so `~/.bashrc` loads `OPENAI_API_KEY`.

## Parse user input

1. **YouTube URL** → `{VIDEO_ID}`
2. **Start** / **End** — `-to` is **absolute** on the full timeline
3. **"till the end"** — `-ss START` only (no `-to`)
4. **Full video** — skip crop; use `~/Downloads/{VIDEO_ID}_full.m4a`
5. **Two-part sermon** — crop each, **merge**, then normalise → transcribe once

## Output naming

| Artifact | Path |
|----------|------|
| Full download | `~/Downloads/{VIDEO_ID}_full.m4a` |
| Cropped / normalised | `~/praisonai-audio-editor/{stem}.m4a` |
| Transcript JSON | `…/{stem}.transcript.json` |
| Plain text | `…/{stem}.transcript.txt` |
| Log | `…/{stem}_transcribe.log` |

## Checklist

```
- [ ] Download (conda yt-dlp)
- [ ] Crop or merge parts
- [ ] Normalise volume — report mean/max before & after
- [ ] ffprobe duration check
- [ ] transcribe --format json
- [ ] extract-text
- [ ] Pack + YAML map (if requested)
```

## Step 1 — Download

```bash
zsh -c "source $(conda info --base)/etc/profile.d/conda.sh && conda activate test && \
  python3 -m yt_dlp -f 'ba/b' --no-playlist --concurrent-fragments 8 --no-part \
  -x --audio-format m4a \
  -o '/Users/praison/Downloads/{VIDEO_ID}_full.%(ext)s' 'URL'"
```

403 → `pip install -U yt-dlp` + `--extractor-args 'youtube:player_client=android,web'`.  
**No `--print` on download** — it can skip writing the file.

## Step 2 — Crop

```bash
ffmpeg -y -nostdin -i FULL.m4a -ss START -to END -map 0:a:0 -c copy OUT.m4a
# end of video: omit -to
```

## Step 2b — Volume optimisation

Target: **−16 LUFS**, **−1.5 dBTP**.

| Condition | Action |
|-----------|--------|
| Too quiet (mean &lt; −22 dB or max &lt; −8 dB) | Auto loudnorm |
| Optimal (−22…−14 mean, −8…−2 max) | Skip — "Volume OK" |
| Too loud (mean &gt; −12 dB or max ≥ −1 dB) | `--in-place --force` |

```bash
cd ~/praisonai-audio-editor
python3 -m praisonai_editor normalize "$AUDIO" --in-place
python3 -m praisonai_editor normalize "$AUDIO" --in-place --force   # hot audio
```

Wrapper: `scripts/optimize-audio-volume.sh`

## Step 2c — Merge two parts

```bash
ffmpeg -y -i PART1.m4a -i PART2.m4a \
  -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[out]" -map "[out]" \
  -c:a aac -b:a 192k MERGED.m4a
```

Normalise merged file, then transcribe once.

## Step 3 — Transcribe

**Full local pipeline (silence cut + normalise + transcribe):** [sermon-transcribe](../sermon-transcribe/SKILL.md) — run **all steps** when user provides audio; this YouTube skill covers download + crop + normalise only.

**Transcribe rules (Tamil+English, `--speed 2`, sanity checks):** [sermon-transcribe](../sermon-transcribe/SKILL.md) Steps 6–10.

Mixed Tamil/English: **omit `--language`**. English-dominant: `--language en`.

```bash
bash -lc 'cd ~/praisonai-audio-editor && python3 -m praisonai_editor transcribe \
  FILE.m4a --format json \
  2>&1 | tee STEM_transcribe.log'
```

Optional: `--speed 2` after 5 min quality test (see sermon-transcribe).

## Step 4 — Extract text

```bash
python3 -m praisonai_editor extract-text STEM.transcript.json
```

→ `{stem}.transcript.txt`

## Step 5–6 — YAML map + deck pack

Map stem → `examples/*.yaml` by topic/verses. Pack script:

```bash
python3 /Users/praison/praisonaippt/scripts/build_sermon_pack2.py
```

Output: `~/Downloads/BIC-Sermon-Deck-Pack-2/` + `sermon_video_map.json`

## Do not use

- Homebrew `yt-dlp` alone (403)
- `--download-sections` on long clips
- `transcribe --force-transcribe`

## More detail

Canonical skill: `~/.cursor/skills/youtube-clip-transcribe/SKILL.md` (`reference.md`, `examples.md`)
