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

import contextlib
import fcntl
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

#: Default for ``prune_sessions()``'s ``max_age_seconds`` -- one week. A
#: session is created the moment a caller picks a source file (e.g. Studio's
#: AI Edit panel, see modules/audio_ai_editor.py) and nothing ever calls
#: ``end_session()`` automatically, so an abandoned session (browser reload,
#: closed tab) otherwise lives on this journal forever.
DEFAULT_SESSION_MAX_AGE_SECONDS = 7 * 24 * 3600


def _sessions_root() -> Path:
    """``~/.praisonai/editor/sessions``. Patch in tests to avoid writing under the real home directory."""
    return Path.home() / ".praisonai" / "editor" / "sessions"


def _session_dir(session_id: str) -> Path:
    return _sessions_root() / session_id


def _session_file(session_id: str) -> Path:
    return _session_dir(session_id) / "history.json"


def _lock_file(session_id: str) -> Path:
    return _session_dir(session_id) / ".lock"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def _session_lock(session_id: str) -> Iterator[None]:
    """Hold an exclusive, BLOCKING lock scoped to one session's journal for
    the duration of a read-modify-write op (``record_edit``/``undo``/
    ``redo``/``reset``).

    Studio's worker (deploy/studio_audio_worker.py) calls ``record_edit()``
    from a background process while a user can simultaneously click Undo/
    Redo/Reset from the browser (a separate Flask request/process) -- both
    read-modify-write the same journal file with no other coordination, so
    without this a write from one can silently clobber the other's change
    (lost update).

    Unlike deploy/studio_audio_worker.py's own ``acquire_lock()``
    (``LOCK_EX | LOCK_NB``, designed to skip a whole worker invocation if
    busy), this BLOCKS: callers wait their turn rather than silently no-op,
    since a lost undo click is worse than a slightly delayed one.

    If the session directory does not exist yet, there is nothing on disk to
    race over -- skip locking (and creating anything) so a call against a
    never-started session_id still cleanly hits ``_load_journal() is None``
    and raises/returns exactly as before, with no directory left behind.
    """
    session_dir = _session_dir(session_id)
    if not session_dir.is_dir():
        yield
        return
    lock_path = _lock_file(session_id)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


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
    # Write-to-temp-then-os.replace(), same convention as every other piece
    # of persistent state in this codebase (Studio's modules/audio_ai_jobs.py
    # write_status, modules/video_jobs.py's equivalent): a reader must never
    # be able to open the journal mid-write and see a truncated/partial
    # file. The temp name is unique per writer (pid + a random suffix), not
    # a shared `history.json.tmp` -- two concurrent writers sharing one tmp
    # name could interleave their JSON into the same file before either
    # calls os.replace(), corrupting both.
    tmp_file = journal_file.parent / f"{journal_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp_file.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    os.replace(tmp_file, journal_file)


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
    with _session_lock(session_id):
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
    with _session_lock(session_id):
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
    with _session_lock(session_id):
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


def jump_to(session_id: str, index: int) -> str | None:
    """Move the pointer directly to an arbitrary valid position, in one
    lock/read/write cycle -- built for Studio's history timeline, where a
    user can click any past step and jump straight to it, instead of a
    caller looping ``undo()``/``redo()`` one step at a time (which would
    multiply lock acquisitions for a long chain, for no real benefit).

    ``index`` follows the same convention as the internal ``pointer``
    field: ``-1`` jumps to the original ``source_path`` (like ``reset()``,
    but WITHOUT discarding anything -- the full stack, including any entries
    past the new pointer, is left intact and still redo-able forward), and
    ``0`` to ``len(stack) - 1`` jumps to that stack entry (0-indexed,
    matching each entry's own ``index`` field as returned by ``history()``,
    so a caller can pass one straight through with no translation).

    Only the pointer moves -- the stack itself is never touched, even when
    jumping to a position earlier than entries that stay unreachable except
    via redo/jump; only ``record_edit`` ever truncates, and only relative to
    wherever the pointer sits at the moment of that new recording.

    Returns the path at the new position (the original ``source_path`` for
    ``-1``, or ``stack[index]["path"]`` otherwise), or ``None`` if
    ``session_id`` is unknown OR ``index`` is out of the session's ACTUAL
    current range ``[-1, len(stack) - 1]`` (read fresh under the lock, not a
    stale value) -- both are ordinary 'nothing happened' cases for a caller,
    not errors, matching ``undo``/``redo``'s own "never raises" contract.
    Jumping to the pointer's current position is a legal no-op that still
    returns the correct path.
    """
    with _session_lock(session_id):
        journal = _load_journal(session_id)
        if journal is None:
            return None
        stack = journal["stack"]
        if index < -1 or index > len(stack) - 1:
            return None
        journal["pointer"] = index
        _save_journal(session_id, journal)
        if index == -1:
            return journal["source_path"]
        return stack[index]["path"]


def reset(session_id: str) -> str | None:
    """Jump back to the session's original ``source_path``.

    Discards ALL edit history (the redo tail too, not just future steps).
    Returns the original ``source_path``, or ``None`` if ``session_id`` is
    unknown. Does NOT delete any of the actual output files the discarded
    history entries pointed to — this is pure bookkeeping, callers own
    file lifecycle.
    """
    with _session_lock(session_id):
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


def prune_sessions(max_age_seconds: float | None = None) -> int:
    """Delete on-disk session directories whose journal hasn't been touched
    in over ``max_age_seconds`` (default ``DEFAULT_SESSION_MAX_AGE_SECONDS``,
    one week).

    Mirrors modules/audio_ai_jobs.py's ``JOB_RETENTION``/``prune_jobs()``
    pattern in spirit, but time-based rather than count-based: a session has
    no natural "queue length" the way a job store does (see that module's
    docstring) -- an abandoned session left by a closed browser tab or a
    page reload (nothing ever calls ``end_session()`` for those) can just as
    easily sit for five minutes as five weeks, and count-based retention
    would prune a session someone is actively mid-edit on just because
    enough OTHER sessions were created meanwhile.

    "Live" here means recently touched: a session younger than
    ``max_age_seconds`` (by its journal file's mtime, updated by every
    ``record_edit``/``undo``/``redo``/``reset``/``start_session`` call) is
    never pruned, the equivalent of ``prune_jobs()`` never touching a queued
    or running job. Does not touch any output files a session's entries
    pointed to -- this module never touches media bytes (see the module
    docstring); a pruned session simply becomes unresolvable by its old id.

    Returns how many session directories were removed. A missing sessions
    root (nothing ever started) is not an error -- just nothing to prune.
    """
    root = _sessions_root()
    if not root.is_dir():
        return 0
    if max_age_seconds is None:
        max_age_seconds = DEFAULT_SESSION_MAX_AGE_SECONDS

    now = time.time()
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        journal_file = entry / "history.json"
        try:
            mtime = journal_file.stat().st_mtime
        except OSError:
            # No readable journal (e.g. a stray/incomplete directory) --
            # fall back to the directory's own mtime rather than skipping it
            # forever.
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
        if now - mtime <= max_age_seconds:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed
