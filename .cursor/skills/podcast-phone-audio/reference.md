# podcast-phone-audio — reference

## ffmpeg / ffprobe

Prefer conda (paired, smoke-tested in `praisonai_editor/probe.py`):

```bash
FF=~/miniconda3/bin/ffmpeg
FP=~/miniconda3/bin/ffprobe
```

Homebrew `/opt/homebrew/bin/ffprobe` may fail with missing `libx265` dylib.

## Noise QA (measure before denoise)

Analyse background noise with per-window `astats` (RMS trough, noise floor) and
`silencedetect`, then pick `afftdn` `nr`/`nf` automatically:

```bash
# Analyse only — prints level (clean/mild/moderate/heavy) + recommendation
python3 -m praisonai_editor noise-qa INPUT --json

# Apply recommended denoise (remuxes video with -c:v copy when present)
python3 -m praisonai_editor noise-qa INPUT --apply -o OUTPUT

# Full per-window report
python3 -m praisonai_editor noise-qa INPUT --report /tmp/noise_qa.json
```

| Level | Typical noise floor | Action |
|-------|---------------------|--------|
| clean | ≤ −52 dB, SNR ≥ 28 dB | Skip denoise |
| mild | −52 to −45 dB | `nr=10`, `nf` ≈ floor + 8 dB |
| moderate | −45 to −35 dB | `nr=14`, `nf` ≈ floor + 5 dB, optional highpass 80 Hz |
| heavy | > −35 dB or low SNR | `nr=20`, highpass 80 Hz |

Best practice: **denoise before normalise** — loudnorm lifts the noise floor.

### Mobile / environmental noise (constant room bed)

Typical phone video: **0% silence**, noise floor **-35 to -40 dB**, default `afftdn nf=-50` **does nothing**.

| Problem | Fix |
|---------|-----|
| `tn=1` with no silent gaps | Use **`tn=0`** (static profile) |
| `nf=-50` vs actual floor -37 dB | Set **`nf` ≈ measured floor + 2 dB** |
| NR=14 too weak | Use **double pass**: `nr=28` then `nr=10` + **highpass 120 Hz** |
| Normalise after denoise | **Skips** if mean ≥ -22 dB — loudnorm lifts room noise |
| Compressor makeup=2 | Use **makeup=1** — less noise lift |

```bash
python3 -m praisonai_editor noise-qa INPUT --json   # read noise_floor_db, silence_ratio
# Pipeline auto-selects mobile path when silence_ratio < 5% and floor > -42 dB
```

**Prefer natural speech over silence:** heavy mobile denoise (`nr=28` double-pass) can sound **artificial/watery**. Default to standard NR=14 unless the user explicitly asks for stronger noise removal. When A/B testing, keep the lighter version if the heavy pass sounds processed.

## Noise reduction (manual)

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

## Crop-only mode

For `.WAV` direct recordings (interface/recorder — no phone memo artefacts):

```bash
process_phone_podcast.sh "$SRC" "$STEM" "$START" "$END" --crop-only
```

Output: `{STEM}_FINAL.m4a` = cropped AAC at 192 kbps. No denoise/EQ/silence/normalise.

### Optional normalise after crop-only

Direct recordings are often quiet (e.g. mean −45 dB). Normalise for Spotify without denoise/EQ:

```bash
cd ~/praisonai-audio-editor
python3 -m praisonai_editor normalize "${STEM}_cropped.m4a" -o "${STEM}_norm.m4a" --force
ln -sf "$(pwd)/${STEM}_norm.m4a" "${STEM}_FINAL.m4a"
```

Target: −16 LUFS. Re-run transcribe only if normalised file replaces FINAL before upload.

## Full pipeline (SDK)

When the user says **full pipeline**, run audio steps **and** biblerevelation publish (Phases 5–11).

```bash
# List all phases (audio → article → Spotify notes)
python3 -m praisonai_editor full-pipeline --describe

# Check which artefacts exist for a stem
python3 -m praisonai_editor full-pipeline bf4fbafd_message --json
```

Implementation: `praisonai_editor/full_pipeline.py`  
Article phases: `~/.cursor/skills/biblerevelation-sermon-articles/pipeline.md`

## Related skills

- [sermon-transcribe](../sermon-transcribe/SKILL.md) — general local audio (no phone denoise/EQ)
- [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) — YouTube download + crop
- `~/.cursor/skills/biblerevelation-sermon-articles/` — article pipeline (**required for full pipeline**)
