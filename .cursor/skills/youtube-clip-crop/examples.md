# YouTube clip crop — examples

## Single URL, multiple ranges

**User input:**
```
https://www.youtube.com/watch?v=f0cDrN8ZwVc
0:50 to 25:50
37:27 to 41:08
1:16:10 to 1:23:31
2:15:25 to 2:20:13
```

**Output files:**
- `f0cDrN8ZwVc_clip1_0050-2550.m4a` — 25:00
- `f0cDrN8ZwVc_clip2_3727-4108.m4a` — 3:41
- `f0cDrN8ZwVc_clip3_7610-8331.m4a` — 7:21
- `f0cDrN8ZwVc_clip4_81525-82013.m4a` — 4:48

**Reference table:**

| File name | URL | From | To | Duration |
|-----------|-----|------|-----|----------|
| `f0cDrN8ZwVc_clip1_0050-2550.m4a` | https://www.youtube.com/watch?v=f0cDrN8ZwVc | 0:50 | 25:50 | 25:00 |
| `f0cDrN8ZwVc_clip2_3727-4108.m4a` | https://www.youtube.com/watch?v=f0cDrN8ZwVc | 37:27 | 41:08 | 3:41 |
| `f0cDrN8ZwVc_clip3_7610-8331.m4a` | https://www.youtube.com/watch?v=f0cDrN8ZwVc | 1:16:10 | 1:23:31 | 7:21 |
| `f0cDrN8ZwVc_clip4_81525-82013.m4a` | https://www.youtube.com/watch?v=f0cDrN8ZwVc | 2:15:25 | 2:20:13 | 4:48 |

---

## Shorthand hour notation

**User input:**
```
https://www.youtube.com/watch?v=jfSN0rEkd9k
11:43 to 47:28
1:8:53 to 1:12:09
1:26:54 to 1:29:50
```

Parse `1:8:53` as `1:08:53`.

**Output files:**
- `jfSN0rEkd9k_clip1_1143-4728.m4a` — 35:45
- `jfSN0rEkd9k_clip2_6853-7209.m4a` — 3:16
- `jfSN0rEkd9k_clip3_8654-8950.m4a` — 2:56

---

## Crop only (explicit)

**User:** "no transcription or articles — only crop and give me audio"

→ Follow this skill only. Skip transcribe, extract-text, normalise, and pack steps.

---

## Re-download skip

If `~/Downloads/jfSN0rEkd9k_full.m4a` already exists from a prior run, go straight to Step 2 (crop).

---

## Webm fallback

If yt-dlp downloads `{VIDEO_ID}_full.webm` but post-processing fails (`ffprobe and ffmpeg not found`), convert then crop:

```bash
ffmpeg -loglevel error -y -nostdin -i ~/Downloads/xfEk4bC35-0_full.webm \
  -c:a aac -b:a 192k ~/Downloads/xfEk4bC35-0_full.m4a
```

Always pass `--ffmpeg-location "$(command -v ffmpeg)"` to yt-dlp to avoid this.

---

## Validate clips

```bash
cd ~/praisonai-audio-editor
./scripts/validate-youtube-clips.sh \
  "42:15" "1:44:03" xfEk4bC35-0_clip1_4215-10403.m4a
```
