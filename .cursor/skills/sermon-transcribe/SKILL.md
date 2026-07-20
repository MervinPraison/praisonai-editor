---
name: sermon-transcribe
description: Transcribes Tamil and English sermon audio (wav/m4a) with praisonai-audio-editor — language auto-detect, optional 2× speed for API cost, extract-text, sanity checks. Use when transcribing sermon clips, mixed Tamil/English audio, autoedited wav, or after crop/normalise; not for YouTube download (see youtube-clip-transcribe).
---

# Sermon audio → transcribe → plain text

Run on the user's Mac. **Do not stop at instructions.**

For **YouTube URL + crop + normalise**, run [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md) first, then use **Step 3 here** for transcribe rules.

## Prerequisites

| Tool | Check |
|------|--------|
| `OPENAI_API_KEY` | `bash -lc 'test -n "$OPENAI_API_KEY" && echo ok'` |
| Repo | `~/praisonai-audio-editor` |
| `ffmpeg` / `ffprobe` | `ffmpeg -version` |

**Shell:** `bash -lc '…'` so `~/.bashrc` loads the API key.

## Input

- Local **`.wav` / `.m4a`** (cropped, autoedited, or normalised)
- Typical stem: `{topic}.wav` or `{VIDEO_ID}_{start}_to_{end}.m4a`

## Checklist

```
- [ ] Choose language mode (mixed Tamil+English → omit --language)
- [ ] Optional: test --speed 2 on first 5 min before full file
- [ ] transcribe --format json
- [ ] extract-text → .transcript.txt
- [ ] Sanity: duration ≈ ffprobe; word count; opening/closing; English in Latin script
- [ ] If 2× quality fails → re-run 1× auto-detect
```

## Step 1 — Language mode

| Audio | Command |
|-------|---------|
| **Tamil sermon + English verses/quotes** | **Omit `--language`** (auto-detect) |
| Pure Tamil | `--language ta` |
| English-dominant | `--language en` |

**Never** `--language ta` on mixed content — English becomes garbled (`opera`, `articles`, transliteration).

Whisper does **not** label Tamil vs English per word; output is one continuous string (fine for `.txt`).

## Step 2 — Speed (optional cost save)

OpenAI `whisper-1` ≈ **$0.006/min** of audio sent. `--speed 2` halves billed minutes; timestamps are scaled back in JSON.

**Quality gate (required):**

1. Transcribe **first 5 min** at `--speed 2`
2. If text is repetitive, &lt; ~500 chars, or Tamil garbled → use **1×** for full file
3. Galatians-style Tamil sermons: **1× auto-detect** is the reliable default

```bash
# Full file with built-in speed (timestamps auto-scaled)
python3 -m praisonai_editor transcribe "$AUDIO" --format json --speed 2 \
  -o "${STEM}.transcript.json"
```

Manual ffmpeg 2× (legacy / debug):

```bash
ffmpeg -y -nostdin -i "$AUDIO" -af "atempo=2.0" -ar 16000 -ac 1 /tmp/sermon_2x.wav
```

## Step 3 — Transcribe (default: 1× auto-detect)

```bash
bash -lc 'cd ~/praisonai-audio-editor && \
  python3 -m praisonai_editor transcribe "$AUDIO" --format json \
    -o "${STEM}.transcript.json" \
    2>&1 | tee "${STEM}_transcribe.log"'
```

Long files: ~600 s chunks automatically. No `--force-transcribe` (flag does not exist).

## Step 4 — Extract plain text

```bash
cd ~/praisonai-audio-editor
python3 -m praisonai_editor extract-text "${STEM}.transcript.json"
```

Output: `{stem}.transcript.txt`

## Step 5 — Sanity check

```bash
ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$AUDIO"
```

```python
import json, re
d = json.load(open("STEM.transcript.json"))
print("dur", d["duration"], "words", len(d["words"]), "len", len(d["text"]))
print("open:", d["text"][:200])
print("close:", d["text"][-200:])
print("english:", re.findall(r"[A-Za-z]{4,}", d["text"])[:12])
```

| Check | Pass |
|-------|------|
| `duration` ≈ ffprobe (±2 s) | ✓ |
| Word count >> 100 for 30 min sermon | ✓ |
| English scripture in Latin script (`Galatians`, `faith`) | ✓ |
| No long repeated phrases | ✓ |

## Outputs

| File | Purpose |
|------|---------|
| `{stem}.transcript.json` | Words + timestamps |
| `{stem}.transcript.txt` | Plain text |
| `{stem}_transcribe.log` | ASR log |

## Do not use

| Action | Why |
|--------|-----|
| `--language ta` on Tamil+English | Garbles English quotes |
| `--speed 2` without quality test | Tamil often degrades (repetition/hallucination) |
| `transcribe --format txt` then edit | Use JSON + `extract-text` to avoid re-ASR |

## More detail

- Language decision tree, cost table, fallback workflow: [reference.md](reference.md)
- YouTube download + crop + normalise: [youtube-clip-transcribe](../youtube-clip-transcribe/SKILL.md)
