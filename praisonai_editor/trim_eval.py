"""Evaluate a trimmed clip by transcribing head/tail windows plus adjacent context (SDK + CLI ``eval``).

Typical use: check **generated** (trimmed) audio by transcribing the **first few seconds** and
**last few seconds**, plus small bands before/after those windows so boundaries are visible.

Context lets you validate phrasing across boundaries: speech just before the head/after
junction, after the head window, before the tail window, and at the very end of the file.

Transcripts are cached under ``~/.praisonai/editor/eval/…`` keyed by the **parent media
path**, file mtime/size, segment ``ss``/``t``, label, and language/model — not by temp
extract paths — so repeat ``eval`` runs avoid redundant ASR.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .models import TranscriptResult, Word
from .phrase_trim import _editor_cache_root, _norm
from .probe import probe_media
from .render import _find_ffmpeg
from .transcribe import DEFAULT_OPENAI_TRANSCRIPTION_MODEL, transcribe_audio

# Defaults: short head/tail samples suitable for validating trim output.
DEFAULT_HEAD_WINDOW_SEC = 10.0
DEFAULT_TAIL_WINDOW_SEC = 10.0
DEFAULT_HEAD_CONTEXT_BEFORE_SEC = 3.0
DEFAULT_HEAD_CONTEXT_AFTER_SEC = 8.0
DEFAULT_TAIL_CONTEXT_BEFORE_SEC = 8.0
DEFAULT_TAIL_CONTEXT_AFTER_SEC = 3.0

# ``--quick`` / quick_eval preset: minimal ASR cost.
QUICK_HEAD_WINDOW_SEC = 5.0
QUICK_TAIL_WINDOW_SEC = 5.0
QUICK_HEAD_CONTEXT_BEFORE_SEC = 2.0
QUICK_HEAD_CONTEXT_AFTER_SEC = 5.0
QUICK_TAIL_CONTEXT_BEFORE_SEC = 5.0
QUICK_TAIL_CONTEXT_AFTER_SEC = 2.0

DEFAULT_OPENAI_EVAL_JUDGE_MODEL = "gpt-4o-mini"
"""Chat model for :func:`_ai_judge_trim_regions` (override with ``OPENAI_EVAL_JUDGE_MODEL``)."""

_EVAL_CACHE_VER = 1
_META_EVAL_VER = "_praisonai_cache_version"
_META_PARENT = "_praisonai_eval_parent_path"
_META_PARENT_MTIME = "_praisonai_eval_parent_mtime_ns"
_META_PARENT_SIZE = "_praisonai_eval_parent_size"
_META_SS = "_praisonai_eval_ss"
_META_T = "_praisonai_eval_t"
_META_LABEL = "_praisonai_eval_label"
_META_LANG = "_praisonai_eval_language"
_META_MODEL = "_praisonai_eval_model"


def eval_transcript_cache_path(
    parent: Path,
    ss: float,
    t: float,
    label: str,
    language: Optional[str],
    model: Optional[str],
) -> Path:
    """Stable path for one eval segment transcript (under ``~/.praisonai/editor/eval/``)."""
    st = parent.stat()
    key_src = "|".join(
        str(x)
        for x in (
            str(parent.resolve()),
            st.st_mtime_ns,
            st.st_size,
            f"{ss:.9f}",
            f"{t:.9f}",
            label,
            language or "",
            model or "",
        )
    )
    h = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return _editor_cache_root() / "eval" / h[:2] / h / "transcript.json"


def _try_load_eval_cache(
    parent: Path,
    ss: float,
    t: float,
    label: str,
    language: Optional[str],
    model: Optional[str],
) -> Optional[TranscriptResult]:
    path = eval_transcript_cache_path(parent, ss, t, label, language, model)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if raw.get(_META_EVAL_VER) != _EVAL_CACHE_VER:
        return None
    try:
        st = parent.stat()
    except OSError:
        return None
    if raw.get(_META_PARENT) != str(parent.resolve()):
        return None
    if raw.get(_META_PARENT_MTIME) != st.st_mtime_ns or raw.get(_META_PARENT_SIZE) != st.st_size:
        return None
    if raw.get(_META_LABEL) != label:
        return None
    if abs(float(raw.get(_META_SS, -1)) - ss) > 1e-5 or abs(float(raw.get(_META_T, -1)) - t) > 1e-5:
        return None
    if (raw.get(_META_LANG) or "") != (language or ""):
        return None
    if (raw.get(_META_MODEL) or "") != (model or ""):
        return None
    core = {k: v for k, v in raw.items() if not k.startswith("_praisonai_")}
    return TranscriptResult.from_dict(core)


def _write_eval_cache(
    parent: Path,
    ss: float,
    t: float,
    label: str,
    language: Optional[str],
    model: Optional[str],
    tr: TranscriptResult,
) -> None:
    st = parent.stat()
    payload = tr.to_dict()
    payload[_META_EVAL_VER] = _EVAL_CACHE_VER
    payload[_META_PARENT] = str(parent.resolve())
    payload[_META_PARENT_MTIME] = st.st_mtime_ns
    payload[_META_PARENT_SIZE] = st.st_size
    payload[_META_SS] = ss
    payload[_META_T] = t
    payload[_META_LABEL] = label
    payload[_META_LANG] = language or ""
    payload[_META_MODEL] = model or ""
    out = eval_transcript_cache_path(parent, ss, t, label, language, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class TrimEdgeEvalReport:
    """Result of :func:`evaluate_trim_edges`."""

    ok: bool
    duration_sec: float
    head_window_sec: float
    tail_window_sec: float
    head_transcript: str
    tail_transcript: str
    head_before_transcript: str = ""
    head_after_transcript: str = ""
    tail_before_transcript: str = ""
    tail_after_transcript: str = ""
    eval_cache_hits: int = 0
    eval_cache_misses: int = 0
    ai_judge_ran: bool = False
    ai_judge_acceptable: Optional[bool] = None
    ai_judge_reason: str = ""
    asr_backend: str = ""
    asr_model: str = ""
    opening_words_timed: List[dict] = field(default_factory=list)
    closing_words_timed: List[dict] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "duration_sec": self.duration_sec,
            "head_window_sec": self.head_window_sec,
            "tail_window_sec": self.tail_window_sec,
            "head_transcript": self.head_transcript,
            "tail_transcript": self.tail_transcript,
            "head_before_transcript": self.head_before_transcript,
            "head_after_transcript": self.head_after_transcript,
            "tail_before_transcript": self.tail_before_transcript,
            "tail_after_transcript": self.tail_after_transcript,
            "eval_cache_hits": self.eval_cache_hits,
            "eval_cache_misses": self.eval_cache_misses,
            "ai_judge_ran": self.ai_judge_ran,
            "ai_judge_acceptable": self.ai_judge_acceptable,
            "ai_judge_reason": self.ai_judge_reason,
            "asr_backend": self.asr_backend,
            "asr_model": self.asr_model,
            "opening_words_timed": list(self.opening_words_timed),
            "closing_words_timed": list(self.closing_words_timed),
            "failures": list(self.failures),
        }


def _ffmpeg_extract_segment(
    inp: Path, out: Path, *, ss: Optional[float] = None, t: float
) -> None:
    ffmpeg = _find_ffmpeg()
    cmd: List[str] = [ffmpeg, "-y"]
    if ss is not None:
        cmd.extend(["-ss", str(ss)])
    cmd.extend(["-i", str(inp), "-t", str(t), "-c", "copy", str(out)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr or r.stdout}")


def _cache_ss_value(ss: Optional[float]) -> float:
    """Normalise seek start for cache keys (``None`` = from file start)."""
    return 0.0 if ss is None else float(ss)


def _timed_word_dicts(words: List[Word], *, offset_sec: float = 0.0) -> List[dict]:
    """Word-level timings for JSON; ``offset_sec`` shifts segment-relative times to file timeline."""
    out: List[dict] = []
    for w in words:
        out.append(
            {
                "text": w.text,
                "start_sec": round(float(w.start) + offset_sec, 3),
                "end_sec": round(float(w.end) + offset_sec, 3),
            }
        )
    return out


def _transcribe_eval_segment(
    parent: Path,
    out_tmp: Path,
    *,
    ss: Optional[float],
    t: float,
    label: str,
    language: Optional[str],
    model: Optional[str],
    use_local: bool,
    use_cache: bool,
    write_cache: bool,
    force_transcribe: bool,
) -> tuple[TranscriptResult, bool]:
    """Return ``(result, from_cache)``."""
    ck_ss = _cache_ss_value(ss)
    if use_cache and not force_transcribe:
        tr = _try_load_eval_cache(parent, ck_ss, t, label, language, model)
        if tr is not None:
            return tr, True
    _ffmpeg_extract_segment(parent, out_tmp, ss=ss, t=t)
    tr = transcribe_audio(str(out_tmp), use_local=use_local, language=language, model=model)
    if write_cache:
        try:
            _write_eval_cache(parent, ck_ss, t, label, language, model, tr)
        except OSError:
            pass
    return tr, False


def _derive_ai_intents(
    *,
    ai_start_intent: Optional[str],
    ai_end_intent: Optional[str],
    head_contains: Optional[str],
    tail_forbid: Optional[Sequence[str]],
) -> tuple[str, str]:
    """Build editor intent strings for the judge when the user omits explicit intents."""
    si_parts: List[str] = []
    if ai_start_intent:
        si_parts.append(ai_start_intent.strip())
    if head_contains:
        si_parts.append(
            f"The opening should include or clearly align with this phrase (ASR may vary): {head_contains!r}"
        )
    start = "\n".join(si_parts) if si_parts else (
        "The clip opening should match the intended start of the kept sermon (e.g. topic discussion)."
    )

    ei_parts: List[str] = []
    if ai_end_intent:
        ei_parts.append(ai_end_intent.strip())
    if tail_forbid:
        phrases = [p for p in tail_forbid if p and str(p).strip()]
        if phrases:
            ei_parts.append(
                "The closing region must not include excluded content such as: "
                + "; ".join(repr(p) for p in phrases)
            )
    end = "\n".join(ei_parts) if ei_parts else (
        "The clip should end before any excluded closing passage (e.g. a formal prayer) "
        "that was meant to be cut off."
    )
    return start, end


def _ai_judge_trim_regions(
    *,
    head_before: str,
    head_core: str,
    head_after: str,
    tail_before: str,
    tail_core: str,
    tail_after: str,
    start_intent: str,
    end_intent: str,
    model: Optional[str],
) -> tuple[bool, str]:
    """Use OpenAI Chat Completions (JSON) to judge whether transcripts match trim intent."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False, "OPENAI_API_KEY is not set"

    from openai import OpenAI

    opening = "\n".join(
        f"{label}: {text}"
        for label, text in (
            ("head_before", head_before),
            ("head_core", head_core),
            ("head_after", head_after),
        )
        if text.strip()
    )
    closing = "\n".join(
        f"{label}: {text}"
        for label, text in (
            ("tail_before", tail_before),
            ("tail_core", tail_core),
            ("tail_after", tail_after),
        )
        if text.strip()
    )

    m = model or os.environ.get("OPENAI_EVAL_JUDGE_MODEL", DEFAULT_OPENAI_EVAL_JUDGE_MODEL)
    system = (
        "You evaluate whether a trimmed audio clip matches the editor's goals, using only "
        "ASR transcripts (you cannot hear audio). Be strict about excluded closing content "
        "(prayers, phrases that should have been cut). Accept minor ASR word errors. "
        "Reply with JSON only: {\"acceptable\": boolean, \"reason\": string}."
    )
    user = (
        f"START INTENT:\n{start_intent}\n\nEND INTENT:\n{end_intent}\n\n"
        f"--- OPENING TRANSCRIPTS ---\n{opening or '(empty)'}\n\n"
        f"--- CLOSING TRANSCRIPTS ---\n{closing or '(empty)'}\n"
    )

    client = OpenAI(api_key=key, timeout=120.0)
    try:
        resp = client.chat.completions.create(
            model=m,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        ok = bool(data.get("acceptable"))
        reason = str(data.get("reason", "")).strip() or "(no reason given)"
        return ok, reason
    except Exception as exc:
        return False, f"ai judge request failed: {exc}"


def evaluate_trim_edges(
    media_path: str | Path,
    *,
    head_window_sec: float = DEFAULT_HEAD_WINDOW_SEC,
    tail_window_sec: float = DEFAULT_TAIL_WINDOW_SEC,
    head_context_before_sec: float = DEFAULT_HEAD_CONTEXT_BEFORE_SEC,
    head_context_after_sec: float = DEFAULT_HEAD_CONTEXT_AFTER_SEC,
    tail_context_before_sec: float = DEFAULT_TAIL_CONTEXT_BEFORE_SEC,
    tail_context_after_sec: float = DEFAULT_TAIL_CONTEXT_AFTER_SEC,
    head_contains: Optional[str] = None,
    tail_contains: Optional[str] = None,
    tail_forbid: Optional[Sequence[str]] = None,
    language: Optional[str] = None,
    use_local: bool = False,
    model: Optional[str] = None,
    use_eval_cache: bool = True,
    write_eval_cache: bool = True,
    force_transcribe: bool = False,
    ai_judge: bool = False,
    ai_start_intent: Optional[str] = None,
    ai_end_intent: Optional[str] = None,
    ai_judge_model: Optional[str] = None,
    include_word_timings: bool = True,
    word_timing_limit: int = 40,
    quiet: bool = False,
) -> TrimEdgeEvalReport:
    """Transcribe the head and tail of ``media_path``, plus **context** bands:

    * **head_before** — last ``head_context_before_sec`` seconds *inside* ``[0, head_window_sec]``
      (speech immediately before the head/after boundary).
    * **head_after** — ``head_window_sec`` … ``head_window_sec + head_context_after_sec``.
    * **tail_before** — audio ending where the tail window starts (lead-in to the closing block).
    * **tail_after** — last ``tail_context_after_sec`` seconds of the file (closing words).

    Checks use merged normalised text: ``head_before + head + head_after`` for ``head_contains``;
    ``tail_before + tail + tail_after`` for ``tail_contains`` / ``tail_forbid`` (so phrases
    split across a boundary are still detected).

    Set any ``*_context_*`` to ``0`` to skip that extract (and leave the matching transcript field blank).

    Segment transcripts are cached under ``~/.praisonai/editor/eval/…`` keyed by the parent file
    path, mtime, size, segment bounds, label, and language/model. Use ``force_transcribe=True``
    or ``use_eval_cache=False`` to ignore the cache; ``write_eval_cache=False`` skips writing.

    When ``ai_judge`` is True, an OpenAI chat model reads the opening/closing transcript bundles
    and ``ai_start_intent`` / ``ai_end_intent`` (or values derived from substring checks) and
    returns JSON ``acceptable`` / ``reason``. Requires ``OPENAI_API_KEY``. Model: ``ai_judge_model``
    or env ``OPENAI_EVAL_JUDGE_MODEL`` (default ``gpt-4o-mini``).

    By default, **word-level times** (from the same ASR as the head/tail windows) are included:
    ``opening_words_timed`` (first words of the file) and ``closing_words_timed`` (last words),
    with ``start_sec`` / ``end_sec`` on the **trimmed file** timeline. Use OpenAI Whisper API
    (``use_local=False``, default) for reliable word timestamps; local ASR may return fewer words.
    Set ``include_word_timings=False`` to omit.
    """
    inp = Path(media_path)
    if not inp.is_file():
        raise FileNotFoundError(str(inp))

    probe = probe_media(str(inp))
    duration = float(probe.duration or 0.0)
    if duration <= 0:
        raise RuntimeError("Could not read media duration")

    h = min(float(head_window_sec), duration)
    tail = min(float(tail_window_sec), duration)
    ss_tail = max(0.0, duration - tail)

    hb = max(0.0, float(head_context_before_sec))
    ha = max(0.0, float(head_context_after_sec))
    tb = max(0.0, float(tail_context_before_sec))
    ta = max(0.0, float(tail_context_after_sec))

    head_before_text = ""
    head_after_text = ""
    tail_before_text = ""
    tail_after_text = ""
    head_text = ""
    tail_text = ""

    failures: List[str] = []
    cache_hits = 0
    cache_misses = 0

    uc = use_eval_cache
    wc = write_eval_cache
    ft = force_transcribe

    with tempfile.TemporaryDirectory(prefix="praisonai_trim_eval_") as td:
        tdir = Path(td)

        tr_head, hit = _transcribe_eval_segment(
            inp,
            tdir / "head_segment.mp3",
            ss=None,
            t=h,
            label="head",
            language=language,
            model=model,
            use_local=use_local,
            use_cache=uc,
            write_cache=wc,
            force_transcribe=ft,
        )
        cache_hits += int(hit)
        cache_misses += int(not hit)
        head_text = (tr_head.text or "").strip()

        tr_tail, hit = _transcribe_eval_segment(
            inp,
            tdir / "tail_segment.mp3",
            ss=ss_tail,
            t=tail,
            label="tail",
            language=language,
            model=model,
            use_local=use_local,
            use_cache=uc,
            write_cache=wc,
            force_transcribe=ft,
        )
        cache_hits += int(hit)
        cache_misses += int(not hit)
        tail_text = (tr_tail.text or "").strip()

        if hb > 0 and h > 0:
            edge_lo = max(0.0, h - min(hb, h))
            edge_len = h - edge_lo
            if edge_len > 0:
                tr_e, hit = _transcribe_eval_segment(
                    inp,
                    tdir / "head_before_edge.mp3",
                    ss=edge_lo,
                    t=edge_len,
                    label="head_before",
                    language=language,
                    model=model,
                    use_local=use_local,
                    use_cache=uc,
                    write_cache=wc,
                    force_transcribe=ft,
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
                head_before_text = (tr_e.text or "").strip()

        if ha > 0 and h < duration:
            aft = min(ha, duration - h)
            if aft > 0:
                tr_ha, hit = _transcribe_eval_segment(
                    inp,
                    tdir / "head_after.mp3",
                    ss=h,
                    t=aft,
                    label="head_after",
                    language=language,
                    model=model,
                    use_local=use_local,
                    use_cache=uc,
                    write_cache=wc,
                    force_transcribe=ft,
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
                head_after_text = (tr_ha.text or "").strip()

        if tb > 0 and ss_tail > 0:
            tb_start = max(0.0, ss_tail - tb)
            tb_len = ss_tail - tb_start
            if tb_len > 0:
                tr_tb, hit = _transcribe_eval_segment(
                    inp,
                    tdir / "tail_before.mp3",
                    ss=tb_start,
                    t=tb_len,
                    label="tail_before",
                    language=language,
                    model=model,
                    use_local=use_local,
                    use_cache=uc,
                    write_cache=wc,
                    force_transcribe=ft,
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
                tail_before_text = (tr_tb.text or "").strip()

        if ta > 0 and tail > 0:
            suf_len = min(ta, tail)
            if suf_len > 0:
                ss_suf = duration - suf_len
                tr_ts, hit = _transcribe_eval_segment(
                    inp,
                    tdir / "tail_after_suffix.mp3",
                    ss=ss_suf,
                    t=suf_len,
                    label="tail_after",
                    language=language,
                    model=model,
                    use_local=use_local,
                    use_cache=uc,
                    write_cache=wc,
                    force_transcribe=ft,
                )
                cache_hits += int(hit)
                cache_misses += int(not hit)
                tail_after_text = (tr_ts.text or "").strip()

    if cache_hits or cache_misses:
        print(
            f"Eval transcription: {cache_misses} segment(s) transcribed, {cache_hits} from cache",
            flush=True,
        )

    limit_w = max(1, int(word_timing_limit))
    opening_words_timed: List[dict] = []
    closing_words_timed: List[dict] = []
    asr_backend = "local" if use_local else "openai"
    asr_model = (model or "").strip() or (
        DEFAULT_OPENAI_TRANSCRIPTION_MODEL if not use_local else "faster-whisper"
    )
    if include_word_timings:
        if tr_head.words:
            opening_words_timed = _timed_word_dicts(tr_head.words[:limit_w], offset_sec=0.0)
        if tr_tail.words:
            closing_words_timed = _timed_word_dicts(tr_tail.words[-limit_w:], offset_sec=ss_tail)
        if not quiet and (opening_words_timed or closing_words_timed):
            if opening_words_timed:
                ow = opening_words_timed
                print(
                    "Opening words (timed): "
                    + " ".join(f"{w['text']} [{w['start_sec']:.2f}-{w['end_sec']:.2f}s]" for w in ow[:12])
                    + (" …" if len(ow) > 12 else ""),
                    flush=True,
                )
            if closing_words_timed:
                cw = closing_words_timed
                print(
                    "Closing words (timed): "
                    + " ".join(f"{w['text']} [{w['start_sec']:.2f}-{w['end_sec']:.2f}s]" for w in cw[-12:])
                    + (" …" if len(cw) > 12 else ""),
                    flush=True,
                )

    head_for_check = _norm(
        " ".join(x for x in (head_before_text, head_text, head_after_text) if x).strip()
    )
    tail_for_check = _norm(
        " ".join(x for x in (tail_before_text, tail_text, tail_after_text) if x).strip()
    )

    if head_contains is not None:
        needle = _norm(head_contains)
        if needle and needle not in head_for_check:
            failures.append(
                f"head (before + core + after): expected substring not found (after normalisation): {head_contains!r}"
            )

    if tail_contains is not None:
        needle = _norm(tail_contains)
        if needle and needle not in tail_for_check:
            failures.append(
                f"tail (before + core + after): expected substring not found (after normalisation): {tail_contains!r}"
            )

    if tail_forbid:
        for phrase in tail_forbid:
            if not phrase or not phrase.strip():
                continue
            bad = _norm(phrase)
            if bad and bad in tail_for_check:
                failures.append(
                    f"tail (before + core + after): forbidden substring present (after normalisation): {phrase!r}"
                )

    ai_ran = False
    ai_ok: Optional[bool] = None
    ai_reason = ""
    if ai_judge:
        si, ei = _derive_ai_intents(
            ai_start_intent=ai_start_intent,
            ai_end_intent=ai_end_intent,
            head_contains=head_contains,
            tail_forbid=tail_forbid,
        )
        ai_ok, ai_reason = _ai_judge_trim_regions(
            head_before=head_before_text,
            head_core=head_text,
            head_after=head_after_text,
            tail_before=tail_before_text,
            tail_core=tail_text,
            tail_after=tail_after_text,
            start_intent=si,
            end_intent=ei,
            model=ai_judge_model,
        )
        ai_ran = True
        if not ai_ok:
            failures.append(f"ai judge: {ai_reason}")

    ok = len(failures) == 0
    return TrimEdgeEvalReport(
        ok=ok,
        duration_sec=duration,
        head_window_sec=h,
        tail_window_sec=tail,
        head_transcript=head_text,
        tail_transcript=tail_text,
        head_before_transcript=head_before_text,
        head_after_transcript=head_after_text,
        tail_before_transcript=tail_before_text,
        tail_after_transcript=tail_after_text,
        eval_cache_hits=cache_hits,
        eval_cache_misses=cache_misses,
        ai_judge_ran=ai_ran,
        ai_judge_acceptable=ai_ok if ai_ran else None,
        ai_judge_reason=ai_reason,
        asr_backend=asr_backend,
        asr_model=asr_model,
        opening_words_timed=opening_words_timed,
        closing_words_timed=closing_words_timed,
        failures=failures,
    )
