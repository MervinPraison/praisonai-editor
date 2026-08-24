"""Undo/redo history for chained edits over a single media file.

Every other module in this package (``pipeline.py``, ``remove_ranges.py``,
``normalize.py``, ``master.py``, ``phrase_trim.py``, ...) is a stateless
function: give it an ``input_path`` and an ``output_path`` and it does one
edit. Nothing in the package remembers that a chain of edits happened, or
lets a caller step back to an earlier version.

This module adds that memory as a thin, ADDITIVE layer on top: a "session"
tracks one ``source_path`` plus an ordered stack of edits applied to it
(each edit is just bookkeeping — ``operation``/``params``/``output_path``;
this module never runs ffmpeg or touches media bytes itself, callers still
call ``remove_time_ranges``/``optimize_audio_volume``/etc. themselves and
then ``record_edit()`` the result). ``undo``/``redo`` move a pointer through
that stack; a fresh ``record_edit`` after an ``undo`` discards the
now-abandoned "redo" tail, exactly like a text editor's undo stack.

State lives entirely on disk, one JSON journal per session, under
``~/.praisonai/editor/sessions/<session_id>/history.json`` — the same
``~/.praisonai/editor/...`` convention used by the transcript cache
(``phrase_trim.py``) and the Demucs stem cache (``_demix.py``). Every public
function re-reads the journal from disk before acting, so a session can be
started in one process (e.g. a CLI invocation) and stepped through by a
completely different one (e.g. a background worker), and two calls in the
same test never share in-memory state.

Usage:
    from praisonai_editor.session import start_session, record_edit, undo, redo
    from praisonai_editor import remove_time_ranges

    sid = start_session("talk.mp3")
    out1 = remove_time_ranges("talk.mp3", ["11:53-12:43"], output_path="talk_cut1.mp3")
    record_edit(sid, "remove_time_ranges", {"ranges": ["11:53-12:43"]}, out1)
    ...
    undo(sid)   # -> back to "talk.mp3"
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sessions_root() -> Path:
    """``~/.praisonai/editor/sessions``. Patch in tests to avoid writing under the real home directory."""
    return Path.home() / ".praisonai" / "editor" / "sessions"


def _session_dir(session_id: str) -> Path:
    return _sessions_root() / session_id


def _session_file(session_id: str) -> Path:
    return _session_dir(session_id) / "history.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_journal(session_id: str) -> dict[str, Any] | None:
    """Read and parse a session's journal, or ``None`` if missing/corrupt."""
    journal_file = _session_file(session_id)
    if not journal_file.is_file():
        return None
    try:
        data = json.loads(journal_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or "source_path" not in data or "stack" not in data:
        return None
    return data


def _save_journal(session_id: str, journal: dict[str, Any]) -> None:
    journal_file = _session_file(session_id)
    journal_file.parent.mkdir(parents=True, exist_ok=True)
    journal_file.write_text(json.dumps(journal, indent=2), encoding="utf-8")


def start_session(source_path: str, session_id: str | None = None) -> str:
    """Begin a new edit session for ``source_path``.

    Returns the session id (a fresh uuid4 hex if none given). If
    ``session_id`` already exists, this RESETS that session back to just
    the original ``source_path``, discarding any history it had.
    """
    sid = session_id or uuid.uuid4().hex
    journal: dict[str, Any] = {
        "source_path": source_path,
        "created_at": _now_iso(),
        "stack": [],
        "pointer": -1,
    }
    _save_journal(sid, journal)
    return sid


def record_edit(session_id: str, operation: str, params: dict, output_path: str) -> dict:
    """Push a new edit onto the session's stack.

    Truncates any 'redo' tail past the current pointer first (real
    undo/redo semantics).

    Raises:
        FileNotFoundError: If ``session_id`` does not exist — unlike the
            read/step functions below, this one CAN raise, since silently
            no-op-ing a lost edit would be a real correctness bug for a
            caller.
    """
    journal = _load_journal(session_id)
    if journal is None:
        raise FileNotFoundError(session_id)

    pointer = journal["pointer"]
    stack = journal["stack"][: pointer + 1]  # discard abandoned redo tail

    entry = {
        "index": len(stack),
        "operation": operation,
        "params": dict(params),
        "path": output_path,
        "timestamp": _now_iso(),
    }
    stack.append(entry)
    journal["stack"] = stack
    journal["pointer"] = entry["index"]
    _save_journal(session_id, journal)
    return dict(entry)


def current_path(session_id: str) -> str | None:
    """The file the session is currently 'at'.

    The original ``source_path`` if no edits are recorded (or all have
    been undone), else the most recent non-undone entry's path. Returns
    ``None`` if ``session_id`` is unknown — unknown-session is an
    expected, non-exceptional case for a caller that lost track of an old
    session, not an error.
    """
    journal = _load_journal(session_id)
    if journal is None:
        return None
    pointer = journal["pointer"]
    if pointer == -1:
        return journal["source_path"]
    return journal["stack"][pointer]["path"]


def undo(session_id: str) -> str | None:
    """Step back one edit.

    Returns the path to revert to (the previous entry's path, or the
    original ``source_path`` if this undoes the very first edit), or
    ``None`` if there is nothing to undo OR ``session_id`` is unknown
    (both are ordinary 'nothing happened' cases for a caller, not errors —
    this never raises for either).
    """
    journal = _load_journal(session_id)
    if journal is None:
        return None
    pointer = journal["pointer"]
    if pointer == -1:
        return None
    new_pointer = pointer - 1
    journal["pointer"] = new_pointer
    _save_journal(session_id, journal)
    if new_pointer == -1:
        return journal["source_path"]
    return journal["stack"][new_pointer]["path"]


def redo(session_id: str) -> str | None:
    """Re-apply the most recently undone edit.

    Returns its path, or ``None`` if there is nothing to redo or
    ``session_id`` is unknown.
    """
    journal = _load_journal(session_id)
    if journal is None:
        return None
    pointer = journal["pointer"]
    stack = journal["stack"]
    if pointer + 1 >= len(stack):
        return None
    new_pointer = pointer + 1
    journal["pointer"] = new_pointer
    _save_journal(session_id, journal)
    return stack[new_pointer]["path"]


def reset(session_id: str) -> str | None:
    """Jump back to the session's original ``source_path``.

    Discards ALL edit history (the redo tail too, not just future steps).
    Returns the original ``source_path``, or ``None`` if ``session_id`` is
    unknown. Does NOT delete any of the actual output files the discarded
    history entries pointed to — this is pure bookkeeping, callers own
    file lifecycle.
    """
    journal = _load_journal(session_id)
    if journal is None:
        return None
    journal["stack"] = []
    journal["pointer"] = -1
    _save_journal(session_id, journal)
    return journal["source_path"]


def history(session_id: str) -> list[dict] | None:
    """All recorded edits, oldest first.

    Each entry as recorded by ``record_edit()`` plus an ``"active": bool``
    key (``True`` if this entry has not been undone past). Returns
    ``None`` if ``session_id`` is unknown, ``[]`` if the session exists
    but has no edits yet.
    """
    journal = _load_journal(session_id)
    if journal is None:
        return None
    pointer = journal["pointer"]
    return [
        {**entry, "active": entry["index"] <= pointer}
        for entry in journal["stack"]
    ]


def session_exists(session_id: str) -> bool:
    """Return ``True`` if ``session_id`` refers to a live, readable session."""
    return _load_journal(session_id) is not None


def end_session(session_id: str) -> bool:
    """Delete a session's on-disk journal.

    Returns ``True`` if it existed and was removed, ``False`` if it was
    already gone. Does not touch any output files it referenced.
    """
    session_dir = _session_dir(session_id)
    existed = session_dir.is_dir()
    if existed:
        shutil.rmtree(session_dir, ignore_errors=True)
    return existed
