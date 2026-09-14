---
name: podcast-phone-audio
description: Processes iPhone voice memo or noisy local audio for Spotify podcast — crop by start/end times, denoise, phone EQ, silence cut, normalise, transcribe. Use when the user gives local m4a/wav with from/to timestamps and mentions phone memo, podcast, Spotify upload, or audio quality issues. Not for YouTube clips or biblerevelation articles unless explicitly requested.
---

# Phone memo → Spotify podcast audio

Run on the user's Mac. **Execute the full pipeline — do not stop at instructions.**

## When to use

| Use this skill | Use something else |
|----------------|-------------------|
| Local `.m4a` / `.wav` + **start → end** crop | YouTube URL → [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) |
| Phone memo / background noise / boxy "well" sound | Clean studio audio → [sermon-transcribe](../sermon-transcribe/SKILL.md) (lighter pipeline) |
| Spotify upload prep only | Article/YAML/publish → `~/.cursor/skills/biblerevelation-sermon-articles/` |

## User input pattern

```
New Recording 382.m4a
12:40 to 30:08
```

Derive:
- `SRC` — usually `~/Downloads/New Recording {N}.m4a`
- `STEM` — `new_recording_{N}`
- `START` / `END` — absolute times on source timeline (`-to` is absolute, not duration)

## Checklist

```
- [ ] Step 1: Audio pipeline (crop → denoise → phone EQ → silence cut → normalise)
- [ ] Step 2: Transcribe (5-min chunks — never single long API call for >10 min Tamil)
- [ ] Step 3: Sanity check (duration, word count, volume report)
- [ ] Step 4: Deliver paths to user
```

**Do not** run article, YAML, images, or WordPress unless the user asks separately.

## Step 1 — Audio pipeline

Use **conda ffmpeg** (Homebrew ffmpeg may be broken on this Mac):

```bash
bash ~/praisonai-audio-editor/.cursor/skills/podcast-phone-audio/scripts/process_phone_podcast.sh \
  "/Users/praison/Downloads/New Recording 382.m4a" \
  new_recording_382 \
  12:40 \
  30:08
```

Pipeline order:
1. **Crop** — `ffmpeg -ss START -to END` → `{STEM}_cropped.m4a`
2. **Denoise** — `praisonai_editor denoise --noise-reduction 14`
3. **Phone EQ** — highpass 100 Hz, cut 250/380 Hz boxiness, lowshelf, light compressor
4. **Silence cut** — `mac/cut-silence.py` → `{STEM}_phone_opt_ALTERED.m4a`
5. **Normalise** — `praisonai_editor normalize` → `{STEM}_norm.m4a`
6. **FINAL** — `ln -sf …/{STEM}_norm.m4a …/{STEM}_FINAL.m4a`

Report mean/max dB before and after normalise.

## Step 2 — Transcribe

**Always** use chunked transcription for Tamil+English sermons:

```bash
bash ~/praisonai-audio-editor/.cursor/skills/podcast-phone-audio/scripts/transcribe_recording.sh new_recording_382 \
  2>&1 | tee ~/praisonai-audio-editor/new_recording_382_FINAL_transcribe.log
```

Rules:
- **Omit `--language`** (auto-detect Tamil + English scripture)
- **Never** `--language ta` on mixed content
- **Never** `--speed 2` or `--local` for Tamil — quality fails
- Do **not** call `praisonai_editor transcribe` on the full FINAL file for audio >10 min

## Step 3 — Sanity check

| Check | Pass |
|-------|------|
| FINAL duration ≈ (end − start) minus silence cut | ✓ |
| Word count reasonable (~40–70 words/min for Tamil sermon) | ✓ |
| English scripture appears in Latin script in transcript | ✓ |
| No single repeated phrase dominating transcript | ✓ |

## Step 4 — Deliverables

Report to user:

| File | Purpose |
|------|---------|
| `{STEM}_FINAL.m4a` | Upload to Spotify |
| `{STEM}_FINAL.transcript.txt` | Reference / future article |
| `{STEM}_FINAL.transcript.json` | Timestamps |

Optional (only if user asks):
- `.agent/{slug}-spotify.md` — title, description, art path
- Copy to `~/Downloads/BIC-Sermon-Deck-Pack-2/`

## Spotify post-upload (biblerevelation.org)

After user publishes on Spotify:

```bash
ssh root@185.249.73.167 'cd /home/hestiaadmin/web/biblerevelation.org/public_html/wordpress && \
  wp eval "echo br_sermon_run_rss_sync();" --allow-root'
# Then link: br_sermon_link_transcript( SERMON_ID, POST_ID );
```

See [reference.md](reference.md) for ffmpeg fixes, NR tuning, and troubleshooting.
