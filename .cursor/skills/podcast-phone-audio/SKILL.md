---
name: podcast-phone-audio
description: Processes local audio for Spotify podcast — crop by start/end times; full phone-memo fix (denoise, EQ) or crop-only for direct/studio recordings. Use when the user gives local m4a/wav with from/to timestamps, mentions phone memo, direct recording, podcast, Spotify, or audio quality. Not for YouTube clips or biblerevelation articles unless explicitly requested.
---

# Phone memo → Spotify podcast audio

Run on the user's Mac. **Execute the full pipeline — do not stop at instructions.**

## When to use

| Use this skill | Use something else |
|----------------|-------------------|
| Local `.m4a` / `.wav` + **start → end** crop | YouTube URL → [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) |
| Phone memo / background noise / boxy "well" sound | **Default pipeline** (denoise + phone EQ) |
| **Direct / studio recording** — no echo fix needed | **`--crop-only`** mode (crop → FINAL m4a) |
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

### Standard (audio + transcript only)

```
- [ ] Step 1: Audio pipeline (crop → denoise → phone EQ → silence cut → normalise)
- [ ] Step 2: Transcribe (5-min chunks — never single long API call for >10 min Tamil)
- [ ] Step 3: Sanity check (duration, word count, volume report)
- [ ] Step 4: Deliver paths to user
```

### Full pipeline (when user says **full pipeline** — run ALL steps)

```
- [ ] Steps 1–4 above (audio + transcript)
- [ ] Step 5: Transcript inventory → .agent/{slug}-transcript-inventory.md
- [ ] Step 6: Article HTML → .agent/biblerevelation-{slug}.html
- [ ] Step 7: audit_transcript_coverage.py — must PASS
- [ ] Step 8: Visual brief + featured image + Spotify episode art
- [ ] Step 9: Publish to biblerevelation.org (praisonaiwp)
- [ ] Step 10: Live verification (HTTP 200)
- [ ] Step 11: Spotify notes → .agent/{slug}-spotify.md
```

**Full pipeline = audio through live article.** Do not stop at transcript.

SDK: `python3 -m praisonai_editor full-pipeline --describe`  
Validate: `python3 -m praisonai_editor full-pipeline {STEM}`

Article workflow: [biblerevelation-sermon-articles](~/.cursor/skills/biblerevelation-sermon-articles/pipeline.md) (transcript-only path when no YAML deck).

**Do not** skip biblerevelation publish when the user says **full pipeline**.  
For audio-only requests (no “full pipeline”), stop after Step 4.

## Step 1 — Audio pipeline

Use **conda ffmpeg** (Homebrew ffmpeg may be broken on this Mac).

### Mode A — Direct / studio recording (`--crop-only`)

When the user says **direct recording**, **no echo fix**, **no phone memo**, or **just crop**:

```bash
bash ~/praisonai-audio-editor/.cursor/skills/podcast-phone-audio/scripts/process_phone_podcast.sh \
  "/Users/praison/Downloads/150926_054013_Tr1.WAV" \
  recording_150926_tr1 \
  36:00 \
  47:31 \
  --crop-only
```

Steps: **crop only** → `{STEM}_FINAL.m4a` (symlink to cropped AAC). Report duration + volumedetect. Skip denoise, EQ, silence cut unless user asks.

If volume is quiet (mean &lt; −30 dB), run **normalise only** after crop — see [reference.md](reference.md). Do not denoise/EQ unless user asks.

### Mode B — Phone memo (default)

```bash
bash ~/praisonai-audio-editor/.cursor/skills/podcast-phone-audio/scripts/process_phone_podcast.sh \
  "/Users/praison/Downloads/New Recording 382.m4a" \
  new_recording_382 \
  12:40 \
  30:08
```

Pipeline order:
1. **Crop** → `{STEM}_cropped.m4a`
2. **Denoise** — `--noise-reduction 14`
3. **Phone EQ** — highpass, boxy-EQ cuts, compressor
4. **Silence cut** — `mac/cut-silence.py`
5. **Normalise** — `praisonai_editor normalize` → `{STEM}_norm.m4a`
6. **FINAL** — symlink `{STEM}_norm.m4a`

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

Optional (standard pipeline only — **included in full pipeline**):
- `.agent/{slug}-spotify.md` — title, description, art path
- Copy to `~/Downloads/BIC-Sermon-Deck-Pack-2/`

## Spotify post-upload (biblerevelation.org)

After Spotify upload (include article URL in description). RSS sync imports the sermon CPT and **auto-links** to the article:

```bash
ssh root@185.249.73.167 'cd /home/hestiaadmin/web/biblerevelation.org/public_html/wordpress && \
  wp eval "echo br_sermon_run_rss_sync();" --allow-root && \
  wp post meta get POST_ID _br_sermon_id --allow-root'
```

Hourly cron also runs this. Manual `br_sermon_link_transcript()` only if auto-link failed.

See [reference.md](reference.md) for ffmpeg fixes, NR tuning, and troubleshooting.
