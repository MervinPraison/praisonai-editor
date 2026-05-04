"""Trim media using transcript phrase boundaries (via ``transcribe_audio`` + ffmpeg.

**Preferred long-service workflow** (repeat when redoing work): split the file at
half-time with ffmpeg, transcribe **only** the second-half file (much less ASR
cost than the full recording), then run ``trim`` on that half file so word
timestamps match the audio you cut. Sidecar cache (see below) is tied to that
input path. By default transcription uses **OpenAI**
``transcribe.DEFAULT_OPENAI_TRANSCRIPTION_MODEL`` (``whisper-1``); pass
``use_local=True`` or CLI ``--local`` for faster-whisper offline.

**Phrase matching (default):** ASR text is never exact. We only *normalise*
(lowercase, strip punctuation) and search for a *substring* over sliding
windows of words—no embeddings, no audio relisten. That can miss paraphrases
or latch onto the wrong repeat.

**Inclusive / exclusive word anchors:** use CLI ``--trim-boundaries phrase-first``
so the clip **starts** at the first word of ``--start`` (that word is included)
and **ends** just before the first word of ``--end`` (that word is excluded).
The default ``window`` mode keeps the older sliding-window timestamps.

**End phrase timing:** the exclusive cut uses the **start** time of the first
word of ``--end`` (default: **last** occurrence; use ``--end-first`` for the
first). Whisper word starts are often slightly **late** versus audible speech,
so you may still hear the opening of the end phrase unless you pass a small
``--end-guard SEC`` (pulls the cut earlier by ``SEC`` seconds).

**AI refinement (default on for CLI ``trim``):** uses the **OpenAI Python
SDK** (``ChatCompletion`` / JSON mode)—**not** ``praisonaiagents`` or
PraisonAI Agent tools (those are used by ``edit --prompt`` / ``prompt_edit``).
A small chat model reads a transcript window around the fuzzy guess and
returns JSON ``start_sec`` / ``end_sec`` (text-only; it does not hear the
file). Requires ``OPENAI_API_KEY``; disable with ``--no-refine-openai`` or
``refine_with_openai=False``. Model: ``OPENAI_TRIM_REFINE_MODEL`` or ``DEFAULT_OPENAI_CHAT_MODEL`` (``gpt-4o-mini``).

Transcript cache (primary): ``~/.praisonai/editor/{stem}_{sha256}/transcript.json``
(full 64 hex digest so names stay unique). An older folder ``{stem}_{sha12}`` is
renamed to the long form on access when present. A legacy file next to the media
(``{name}.praisonai.transcript.json``) is still read if neither editor cache hits.
Re-used when path, size, and mtime match; use ``--force-transcribe`` or ``--no-cache`` to ignore and re-transcribe (then overwrite cache unless ``--no-cache-write``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .models import TranscriptResult, Word
from .render import _find_ffmpeg
from .transcribe import transcribe_audio

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
"""Default chat model for trim boundary refinement (override with ``OPENAI_TRIM_REFINE_MODEL``)."""

_CACHE_VER = 1
_META_VER = "_praisonai_cache_version"
_META_PATH = "_praisonai_audio_path"
_META_MTIME = "_praisonai_audio_mtime_ns"
_META_SIZE = "_praisonai_audio_size"


def _editor_cache_root() -> Path:
    """``~/.praisonai/editor``. Patch in tests to avoid writing under the real home directory."""
    return Path.home() / ".praisonai" / "editor"


def _safe_media_stem(media_path: Path) -> str:
    stem = re.sub(r"[^\w\-]+", "_", media_path.stem, flags=re.UNICODE).strip("_")
    return (stem or "media")[:80]


def _media_path_digest(media_path: Path) -> str:
    return hashlib.sha256(str(media_path.resolve()).encode("utf-8")).hexdigest()


def _media_cache_dir_name(media_path: Path, *, digest_chars: int | None = None) -> str:
    """Directory name ``{stem}_{hex}``; ``digest_chars`` defaults to full 64 for uniqueness."""
    full = _media_path_digest(media_path)
    n = 64 if digest_chars is None else max(8, min(digest_chars, 64))
    digest = full[:n]
    return f"{_safe_media_stem(media_path)}_{digest}"


def transcript_cache_file(media_path: Path) -> Path:
    """Primary trim transcript cache: ``~/.praisonai/editor/{stem}_{sha256}/transcript.json``."""
    return _editor_cache_root() / _media_cache_dir_name(media_path) / "transcript.json"


def _trim_cache_file_short_digest_legacy(media_path: Path) -> Path:
    """Older layout used a 12-character digest (still read, then folder may be renamed)."""
    return _editor_cache_root() / _media_cache_dir_name(media_path, digest_chars=12) / "transcript.json"


def _upgrade_short_digest_cache_dir(media_path: Path) -> None:
    """Rename ``{stem}_{12hex}`` → ``{stem}_{64hex}`` when only the short layout exists."""
    root = _editor_cache_root()
    old_dir = root / _media_cache_dir_name(media_path, digest_chars=12)
    new_dir = root / _media_cache_dir_name(media_path)
    if not old_dir.is_dir() or new_dir.exists():
        return
    try:
        old_dir.rename(new_dir)
    except OSError:
        return


def transcript_sidecar_path(media_path: Path) -> Path:
    """Legacy path beside the media file (read if ``transcript_cache_file`` is missing)."""
    return media_path.with_name(f"{media_path.name}.praisonai.transcript.json")


def _parse_transcript_cache_file(cache_file: Path, media_path: Path) -> Optional[TranscriptResult]:
    if not cache_file.is_file():
        return None
    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if raw.get(_META_VER) != _CACHE_VER:
        return None
    try:
        st = media_path.stat()
    except OSError:
        return None
    if raw.get(_META_PATH) != str(media_path.resolve()):
        return None
    if raw.get(_META_MTIME) != st.st_mtime_ns or raw.get(_META_SIZE) != st.st_size:
        return None
    core = {k: v for k, v in raw.items() if not k.startswith("_praisonai_")}
    return TranscriptResult.from_dict(core)


def _try_load_transcript_cache(media_path: Path) -> tuple[Optional[TranscriptResult], Optional[Path]]:
    """Return ``(transcript, path_used)`` or ``(None, None)``."""
    _upgrade_short_digest_cache_dir(media_path)

    primary = transcript_cache_file(media_path)
    tr = _parse_transcript_cache_file(primary, media_path)
    if tr is not None:
        return tr, primary

    short_file = _trim_cache_file_short_digest_legacy(media_path)
    tr = _parse_transcript_cache_file(short_file, media_path)
    if tr is not None:
        return tr, short_file

    legacy = transcript_sidecar_path(media_path)
    tr = _parse_transcript_cache_file(legacy, media_path)
    if tr is not None:
        return tr, legacy
    return None, None


def _write_transcript_cache(media_path: Path, tr: TranscriptResult) -> None:
    _upgrade_short_digest_cache_dir(media_path)
    st = media_path.stat()
    payload = tr.to_dict()
    payload[_META_VER] = _CACHE_VER
    payload[_META_PATH] = str(media_path.resolve())
    payload[_META_MTIME] = st.st_mtime_ns
    payload[_META_SIZE] = st.st_size
    out = transcript_cache_file(media_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _first_phrase_start(words: List[Word], phrase_norm: str, max_span: int = 55) -> Optional[float]:
    n = len(words)
    for i in range(n):
        parts: List[str] = []
        for j in range(i, min(i + max_span, n)):
            parts.append(words[j].text)
            if phrase_norm in _norm(" ".join(parts)):
                return float(words[i].start)
    return None


def _refine_bounds_openai(
    words: List[Word],
    duration: float,
    start_hint: str,
    end_hint: str,
    t0: float,
    t1: float,
) -> Tuple[float, float]:
    """Pick start/end seconds via OpenAI Chat Completions (not PraisonAI agents)."""
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI refinement")

    dur = max(duration, 0.01)
    lo = max(0.0, min(t0, t1) - 90.0)
    hi = min(dur, max(t0, t1) + 90.0)
    lines: List[str] = []
    for w in words:
        if w.end < lo or w.start > hi:
            continue
        lines.append(f"{w.start:.3f}\t{w.end:.3f}\t{w.text}")
    chunk = "\n".join(lines[:1200])
    if not chunk.strip():
        chunk = "\n".join(f"{w.start:.3f}\t{w.end:.3f}\t{w.text}" for w in words[:400])

    system = (
        "You choose precise clip boundaries from a word-level transcript. "
        "Each line is: start_sec<TAB>end_sec<TAB>word. "
        "start_sec for the output clip must be the start time of the first word that "
        "opens the user's intended passage (paraphrases and ASR errors allowed). "
        "end_sec must be the start time of the first word of the closing passage that "
        "must be CUT OFF—the clip ends *before* that word (exclusive). "
        "Reply with JSON only: {\"start_sec\": number, \"end_sec\": number} using values from the transcript."
    )
    user = (
        f"Opening intent (clip begins at first matching word): {start_hint!r}\n"
        f"Closing intent (clip ends *before* this phrase begins): {end_hint!r}\n"
        f"Fuzzy search guess: start ~{t0:.3f}s, exclusive end ~{t1:.3f}s.\n"
        f"Approximate media duration: {dur:.1f}s.\n\nTranscript window:\n{chunk}"
    )

    model = os.environ.get("OPENAI_TRIM_REFINE_MODEL", DEFAULT_OPENAI_CHAT_MODEL)
    client = OpenAI(api_key=key, timeout=120.0)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = json.loads(raw)
    a, b = float(data["start_sec"]), float(data["end_sec"])
    a = max(0.0, min(a, dur))
    b = max(0.0, min(b, dur))
    if b <= a:
        raise ValueError("refinement returned end_sec <= start_sec")
    return a, b


def _phrase_match_starts(words: List[Word], phrase_norm: str, max_span: int = 25) -> List[float]:
    n = len(words)
    starts: List[float] = []
    for i in range(n):
        parts: List[str] = []
        for j in range(i, min(i + max_span, n)):
            parts.append(words[j].text)
            if phrase_norm in _norm(" ".join(parts)):
                starts.append(float(words[i].start))
                break
    return starts


def _first_sliding_match_window(
    words: List[Word], phrase_norm: str, max_span: int
) -> Optional[Tuple[int, int]]:
    """First inclusive word indices (i, j) where the normalised join contains ``phrase_norm``."""
    n = len(words)
    for i in range(n):
        parts: List[str] = []
        for j in range(i, min(i + max_span, n)):
            parts.append(words[j].text)
            if phrase_norm in _norm(" ".join(parts)):
                return i, j
    return None


def _tighten_left_to_phrase_first(words: List[Word], i: int, j: int, phrase_norm: str) -> int:
    """Largest index ``ii`` in ``[i, j]`` with the phrase still in ``join(ii..j)`` (first word of phrase)."""
    best_ii = i
    for ii in range(i, j + 1):
        if phrase_norm in _norm(" ".join(words[t].text for t in range(ii, j + 1))):
            best_ii = ii
    return best_ii


def _first_phrase_first_word_time(words: List[Word], phrase_norm: str, max_span: int = 55) -> Optional[float]:
    """Start time of the first word of the first phrase match (inclusive clip start)."""
    win = _first_sliding_match_window(words, phrase_norm, max_span)
    if win is None:
        return None
    lo, hi = win
    ii = _tighten_left_to_phrase_first(words, lo, hi, phrase_norm)
    return float(words[ii].start)


def _exclusive_end_phrase_first_word_time(
    words: List[Word], phrase_norm: str, max_span: int = 25, *, end_last_match: bool
) -> Optional[float]:
    """Exclusive end: start time of the first word of ``--end`` (last or first occurrence)."""
    n = len(words)
    cands: List[float] = []
    for j in range(n):
        i_lo = max(0, j - max_span + 1)
        best_ii: Optional[int] = None
        for ii in range(i_lo, j + 1):
            if phrase_norm in _norm(" ".join(words[t].text for t in range(ii, j + 1))):
                best_ii = ii if best_ii is None else max(best_ii, ii)
        if best_ii is not None:
            cands.append(float(words[best_ii].start))
    if not cands:
        return None
    return max(cands) if end_last_match else min(cands)


def trim_between_phrase_markers(
    input_path: str,
    output_path: str,
    *,
    start_phrase: str,
    end_phrase: str,
    end_last_match: bool = True,
    use_local: bool = False,
    language: Optional[str] = None,
    model: Optional[str] = None,
    transcript_path: Optional[str] = None,
    use_transcript_cache: bool = True,
    write_transcript_cache: bool = True,
    force_transcribe: bool = False,
    refine_with_openai: bool = True,
    end_guard_seconds: float = 0.0,
    trim_boundaries: str = "window",
) -> str:
    """Transcribe (or load cache / explicit JSON), locate phrases, then ffmpeg ``-c copy`` trim.

    With ``trim_boundaries='phrase-first'``, output runs from the first word of
    the matched ``start_phrase`` (inclusive) up to (exclusive) the first word of
    ``end_phrase``. With ``window``, timestamps follow the legacy sliding-window
    rule. When ``end_last_match`` is True, the end phrase uses its **last**
    occurrence unless ``end_first`` is set on the CLI.

    Args:
        use_local: If False (default), use OpenAI ``whisper-1`` via ``transcribe_audio``.
        model: With API, defaults to ``whisper-1``; with ``use_local=True``, faster-whisper size (e.g. ``base``).
        transcript_path: If set, load this JSON as the transcript (no ASR).
        use_transcript_cache: If True and ``transcript_path`` is unset, try
            ``transcript_cache_file`` then the legacy path beside the media.
        write_transcript_cache: After a fresh ASR run, write under
            ``~/.praisonai/editor/…/transcript.json``.
        force_transcribe: Ignore disk transcript cache and run ASR again; cache file is overwritten on success if ``write_transcript_cache`` is True.
        refine_with_openai: After fuzzy word match, call OpenAI chat (unless
            False). Same behaviour as CLI ``--refine-openai`` / ``--no-refine-openai``.
        end_guard_seconds: If > 0, subtract from the exclusive ``t1`` after
            fuzzy match and optional refinement (clamped so the clip stays valid).
        trim_boundaries: ``phrase-first`` — start at the first word of the matched
            ``start_phrase`` (inclusive); end before the first word of ``end_phrase``
            (exclusive). ``window`` — legacy sliding-window left edge (may start
            before the spoken phrase). CLI: ``--trim-boundaries``.

    Returns:
        Absolute path to the written file.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(input_path)

    tr: Optional[TranscriptResult] = None
    if transcript_path:
        tp = Path(transcript_path)
        if not tp.is_file():
            raise FileNotFoundError(transcript_path)
        tr = TranscriptResult.from_dict(json.loads(tp.read_text(encoding="utf-8")))
    elif not force_transcribe and use_transcript_cache:
        tr, cache_path = _try_load_transcript_cache(inp)
        if tr is not None and cache_path is not None:
            print(f"Using cached transcript: {cache_path}", flush=True)

    if tr is None:
        tr = transcribe_audio(
            str(inp),
            use_local=use_local,
            language=language,
            model=model,
        )
        if write_transcript_cache and not transcript_path:
            try:
                _write_transcript_cache(inp, tr)
            except OSError:
                pass
    if not tr.words:
        raise RuntimeError("Transcription returned no word-level timings.")

    if trim_boundaries not in ("window", "phrase-first"):
        raise ValueError("trim_boundaries must be 'window' or 'phrase-first'")

    s_norm = _norm(start_phrase)
    e_norm = _norm(end_phrase)
    if trim_boundaries == "phrase-first":
        t0 = _first_phrase_first_word_time(tr.words, s_norm)
        if t0 is None:
            t0 = _first_phrase_first_word_time(
                tr.words, _norm("what topic are we seeing anyone know about that")
            )
        if t0 is None:
            t0 = _first_phrase_first_word_time(tr.words, _norm("so what topic are we seeing"))
        if t0 is None:
            raise RuntimeError(f"Start phrase not found in transcript: {start_phrase!r}")
        t1_opt = _exclusive_end_phrase_first_word_time(
            tr.words, e_norm, end_last_match=end_last_match
        )
        if t1_opt is None:
            t1 = float(tr.duration or tr.words[-1].end)
        else:
            t1 = t1_opt
    else:
        t0 = _first_phrase_start(tr.words, s_norm)
        if t0 is None:
            t0 = _first_phrase_start(tr.words, _norm("what topic are we seeing anyone know about that"))
        if t0 is None:
            t0 = _first_phrase_start(tr.words, _norm("so what topic are we seeing"))
        if t0 is None:
            raise RuntimeError(f"Start phrase not found in transcript: {start_phrase!r}")

        end_starts = _phrase_match_starts(tr.words, e_norm)
        if not end_starts:
            t1 = float(tr.duration or tr.words[-1].end)
        else:
            t1 = max(end_starts) if end_last_match else min(end_starts)

    if t1 <= t0:
        raise RuntimeError(f"Invalid trim range: start={t0}, end={t1}")

    if refine_with_openai:
        dur = float(tr.duration or (tr.words[-1].end if tr.words else t1))
        try:
            t0, t1 = _refine_bounds_openai(
                tr.words, dur, start_phrase, end_phrase, t0, t1
            )
            print(
                f"OpenAI refinement: start={t0:.3f}s end={t1:.3f}s (exclusive)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"OpenAI refinement skipped, using fuzzy bounds: {exc}",
                flush=True,
            )

    if t1 <= t0:
        raise RuntimeError(f"Invalid trim range after refinement: start={t0}, end={t1}")

    if end_guard_seconds and end_guard_seconds > 0:
        t1 = max(t0 + 0.001, t1 - float(end_guard_seconds))

    if t1 <= t0:
        raise RuntimeError(f"Invalid trim range after end guard: start={t0}, end={t1}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    span = t1 - t0
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(t0),
        "-i",
        str(inp),
        "-t",
        str(span),
        "-c",
        "copy",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr or r.stdout}")
    return str(out.resolve())
