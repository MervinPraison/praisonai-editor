"""Tests for the undo/redo edit-session journal."""

import json
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import praisonai_editor.session as session_mod
from praisonai_editor.session import (
    DEFAULT_SESSION_MAX_AGE_SECONDS,
    current_path,
    end_session,
    history,
    jump_to,
    prune_sessions,
    record_edit,
    redo,
    reset,
    session_exists,
    start_session,
    undo,
)


@pytest.fixture
def sessions_home(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_sessions_root", lambda: root)
    return root


def test_start_session_generates_fresh_id(sessions_home, tmp_path):
    src = str(tmp_path / "a.mp3")
    sid1 = start_session(src)
    sid2 = start_session(src)
    assert sid1 != sid2
    assert session_exists(sid1)
    assert session_exists(sid2)


def test_start_session_accepts_explicit_id(sessions_home, tmp_path):
    src = str(tmp_path / "a.mp3")
    sid = start_session(src, session_id="my-session")
    assert sid == "my-session"
    assert current_path(sid) == src


def test_start_session_resets_existing_session(sessions_home, tmp_path):
    src = str(tmp_path / "a.mp3")
    sid = start_session(src, session_id="reused")
    record_edit(sid, "normalize", {}, str(tmp_path / "out1.mp3"))
    assert current_path(sid) == str(tmp_path / "out1.mp3")

    # Starting again with the same id wipes history back to (a possibly new) source.
    new_src = str(tmp_path / "b.mp3")
    sid2 = start_session(new_src, session_id="reused")
    assert sid2 == "reused"
    assert current_path(sid2) == new_src
    assert history(sid2) == []


def test_record_edit_chains_and_current_path_tracks_latest(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)

    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    out3 = str(tmp_path / "out3.mp3")

    e1 = record_edit(sid, "normalize", {"target_lufs": -16.0}, out1)
    assert e1["index"] == 0
    assert e1["operation"] == "normalize"
    assert e1["params"] == {"target_lufs": -16.0}
    assert e1["path"] == out1
    assert "timestamp" in e1
    assert current_path(sid) == out1

    e2 = record_edit(sid, "remove_time_ranges", {"ranges": ["1:00-1:10"]}, out2)
    assert e2["index"] == 1
    assert current_path(sid) == out2

    e3 = record_edit(sid, "master", {"preset": "speech"}, out3)
    assert e3["index"] == 2
    assert current_path(sid) == out3


def test_record_edit_raises_for_unknown_session(sessions_home, tmp_path):
    with pytest.raises(FileNotFoundError):
        record_edit("does-not-exist", "normalize", {}, str(tmp_path / "out.mp3"))


def test_undo_steps_back_including_to_original_source(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op1", {}, out1)
    record_edit(sid, "op2", {}, out2)

    assert current_path(sid) == out2
    assert undo(sid) == out1
    assert current_path(sid) == out1
    assert undo(sid) == src
    assert current_path(sid) == src
    # Nothing left to undo.
    assert undo(sid) is None
    assert current_path(sid) == src


def test_redo_reapplies_undone_edit(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op1", {}, out1)
    record_edit(sid, "op2", {}, out2)

    undo(sid)
    undo(sid)
    assert current_path(sid) == src

    assert redo(sid) == out1
    assert current_path(sid) == out1
    assert redo(sid) == out2
    assert current_path(sid) == out2
    # Nothing left to redo.
    assert redo(sid) is None
    assert current_path(sid) == out2


def test_record_edit_after_undo_discards_redo_branch(sessions_home, tmp_path):
    """The one subtle semantic bug an implementation could get wrong: a fresh
    edit after an undo must truncate the abandoned redo tail, not just append."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op1", {}, out1)
    record_edit(sid, "op2", {}, out2)

    # Step back to out1 — out2 is now an abandoned "redo" branch.
    assert undo(sid) == out1

    out3 = str(tmp_path / "out3.mp3")
    e3 = record_edit(sid, "op3", {}, out3)
    # The new entry takes over the abandoned slot, not a new index past it.
    assert e3["index"] == 1
    assert current_path(sid) == out3

    # out2 must no longer be reachable via redo.
    assert redo(sid) is None
    assert current_path(sid) == out3

    entries = history(sid)
    assert [e["path"] for e in entries] == [out1, out3]
    assert [e["active"] for e in entries] == [True, True]


def test_reset_discards_everything_back_to_original_source(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    record_edit(sid, "op1", {}, str(tmp_path / "out1.mp3"))
    record_edit(sid, "op2", {}, str(tmp_path / "out2.mp3"))
    undo(sid)  # leave a redo branch too

    result = reset(sid)
    assert result == src
    assert current_path(sid) == src
    assert history(sid) == []
    # The redo branch is gone as well.
    assert redo(sid) is None
    assert undo(sid) is None


def test_unknown_session_id_never_raises_for_read_or_step_functions(sessions_home):
    unknown = "totally-unknown-session-id"
    assert current_path(unknown) is None
    assert undo(unknown) is None
    assert redo(unknown) is None
    assert reset(unknown) is None
    assert history(unknown) is None
    assert session_exists(unknown) is False


# ---------------------------------------------------------------------------
# jump_to
# ---------------------------------------------------------------------------


def test_jump_to_forward_multiple_steps_keeps_redo_tail_intact(sessions_home, tmp_path):
    """The key property that distinguishes jump_to from a buggy loop of
    undo()/redo() calls: jumping straight from near the start to a far
    entry must leave every entry in between still on the stack, not just
    the endpoint."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    outs = [str(tmp_path / f"out{i}.mp3") for i in range(5)]
    for i, out in enumerate(outs):
        record_edit(sid, f"op{i}", {}, out)

    # Step back near the start.
    undo(sid)
    undo(sid)
    undo(sid)
    undo(sid)
    assert current_path(sid) == outs[0]

    # Jump forward multiple steps at once, straight to the last entry.
    assert jump_to(sid, 4) == outs[4]
    assert current_path(sid) == outs[4]

    # The full stack survived untouched.
    entries = history(sid)
    assert [e["path"] for e in entries] == outs
    assert [e["active"] for e in entries] == [True] * 5

    # And it's still fully undo/redo-able afterward -- nothing was wiped.
    assert undo(sid) == outs[3]
    assert redo(sid) == outs[4]


def test_jump_to_negative_one_matches_reset_path_but_preserves_history(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op1", {}, out1)
    record_edit(sid, "op2", {}, out2)

    result = jump_to(sid, -1)
    assert result == src
    assert current_path(sid) == src

    # Unlike reset(), the stack is still fully intact and redo-able.
    entries = history(sid)
    assert [e["path"] for e in entries] == [out1, out2]
    assert [e["active"] for e in entries] == [False, False]
    assert redo(sid) == out1
    assert redo(sid) == out2


def test_jump_to_out_of_range_returns_none(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    record_edit(sid, "op1", {}, out1)

    assert jump_to(sid, 1) is None   # only index 0 exists
    assert jump_to(sid, -2) is None  # -1 is the floor
    # Nothing happened -- pointer unchanged.
    assert current_path(sid) == out1


def test_jump_to_unknown_session_returns_none(sessions_home):
    assert jump_to("totally-unknown-session-id", 0) is None


def test_jump_to_current_position_is_a_harmless_noop(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    out1 = str(tmp_path / "out1.mp3")
    record_edit(sid, "op1", {}, out1)

    # Already at index 0 -- jumping there again is a no-op, not an error.
    assert jump_to(sid, 0) == out1
    assert current_path(sid) == out1

    jump_to(sid, -1)
    # Already at -1 -- jumping there again is a no-op too.
    assert jump_to(sid, -1) == src
    assert current_path(sid) == src


def test_concurrent_jump_to_no_corruption(sessions_home, tmp_path):
    """Multiple threads calling jump_to with different valid indices on the
    same session at once must never corrupt the journal. The exact final
    pointer among the racers is inherently nondeterministic (whichever
    write lands last wins) -- the only hard guarantee this asserts is that
    the journal stays valid, the pointer always lands on a real position,
    and the stack itself (jump_to never touches it) is unchanged."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    n = 6
    outs = [str(tmp_path / f"out{i}.mp3") for i in range(n)]
    for i, out in enumerate(outs):
        record_edit(sid, f"op{i}", {}, out)

    valid_indices = list(range(-1, n)) * 4  # hammer every valid index repeatedly

    def _jump(idx):
        return jump_to(sid, idx)

    with ThreadPoolExecutor(max_workers=len(valid_indices)) as pool:
        results = list(pool.map(_jump, valid_indices))

    # Every call landed on a real position -- none lost to corruption.
    assert all(r is not None for r in results)

    journal_file = sessions_home / sid / "history.json"
    raw = json.loads(journal_file.read_text(encoding="utf-8"))
    assert -1 <= raw["pointer"] < len(raw["stack"])
    assert len(raw["stack"]) == n  # jump_to never touches the stack
    assert [e["path"] for e in raw["stack"]] == outs


def test_history_reports_active_flag_and_oldest_first(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    assert history(sid) == []

    out1 = str(tmp_path / "out1.mp3")
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op1", {"a": 1}, out1)
    record_edit(sid, "op2", {"b": 2}, out2)

    entries = history(sid)
    assert [e["path"] for e in entries] == [out1, out2]
    assert [e["active"] for e in entries] == [True, True]

    undo(sid)
    entries = history(sid)
    assert [e["active"] for e in entries] == [True, False]


def test_session_exists_and_end_session_round_trip(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    assert session_exists(sid) is True

    assert end_session(sid) is True
    assert session_exists(sid) is False
    # Already gone — second call reports False, not an error.
    assert end_session(sid) is False


def test_journal_persists_across_separate_calls_from_disk(sessions_home, tmp_path):
    """Every public function re-reads the journal file rather than relying on any
    in-memory cache, since a real caller integrates this from a separate worker
    process. Prove it by reading the raw JSON on disk between calls instead of
    holding any live session object."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src, session_id="cross-process")
    out1 = str(tmp_path / "out1.mp3")
    record_edit(sid, "op1", {}, out1)

    journal_file = sessions_home / sid / "history.json"
    assert journal_file.is_file()
    raw = json.loads(journal_file.read_text(encoding="utf-8"))
    assert raw["source_path"] == src
    assert raw["pointer"] == 0
    assert raw["stack"][0]["path"] == out1

    # A fresh call sequence (as a second process would make, with nothing held
    # over from the calls above) sees exactly what's on disk.
    assert current_path(sid) == out1
    out2 = str(tmp_path / "out2.mp3")
    record_edit(sid, "op2", {}, out2)

    raw2 = json.loads(journal_file.read_text(encoding="utf-8"))
    assert raw2["pointer"] == 1
    assert raw2["stack"][1]["path"] == out2
    assert current_path(sid) == out2


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_save_journal_writes_atomically_no_tmp_files_left_behind(sessions_home, tmp_path):
    """_save_journal must write via temp-file-then-os.replace(), the same
    convention Studio's modules/audio_ai_jobs.write_status and
    modules/video_jobs.py's equivalent already use, so a reader can never
    open the journal mid-write and see a truncated/partial file. Prove no
    stray .tmp file is left in the session directory after normal writes,
    and that the only other file present is the lock file record_edit takes
    (see _session_lock)."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src, session_id="atomic-check")
    record_edit(sid, "op1", {}, str(tmp_path / "out1.mp3"))
    undo(sid)
    redo(sid)

    session_dir = sessions_home / sid
    names = {p.name for p in session_dir.iterdir()}
    assert "history.json" in names
    assert not any(n.endswith(".tmp") for n in names)
    assert names <= {"history.json", ".lock"}

    # And the content itself must always be valid, complete JSON.
    raw = json.loads((session_dir / "history.json").read_text(encoding="utf-8"))
    assert raw["source_path"] == src


# ---------------------------------------------------------------------------
# Concurrency: record_edit/undo/redo/reset must not lose updates
# ---------------------------------------------------------------------------


def _mp_worker_record_edit(root, session_id, idx):
    """Top-level (picklable) target for the multiprocessing concurrency test
    below. Reimports the session module fresh in the child process and
    points it at the same isolated sessions root via direct attribute
    assignment -- pytest's monkeypatch fixture only patches within the
    parent process, so a spawned child must be repointed independently."""
    import praisonai_editor.session as child_session_mod

    child_session_mod._sessions_root = lambda: root
    child_session_mod.record_edit(session_id, f"op{idx}", {"idx": idx}, f"/tmp/out{idx}.mp3")


def test_concurrent_record_edit_via_threads_no_lost_updates(sessions_home, tmp_path):
    """Studio's worker calls record_edit() from a background process while a
    user can simultaneously click Undo/Redo/Reset from a separate Flask
    request -- both read-modify-write the same journal file with no other
    coordination. Prove the locking added to record_edit actually serializes
    those writes: hammer record_edit from many threads at once (I/O in
    _save_journal releases the GIL, so without real locking this can and
    does interleave) and confirm the final journal has exactly one entry per
    successful call, with unique, contiguous indices -- not a corrupted file
    and not a lost update."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    n = 25

    def _record(i):
        record_edit(sid, f"op{i}", {"i": i}, str(tmp_path / f"out{i}.mp3"))

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(_record, range(n)))

    entries = history(sid)
    assert len(entries) == n
    assert sorted(e["index"] for e in entries) == list(range(n))

    journal_file = sessions_home / sid / "history.json"
    raw = json.loads(journal_file.read_text(encoding="utf-8"))
    assert len(raw["stack"]) == n
    assert raw["pointer"] == n - 1


def test_concurrent_record_edit_via_processes_no_lost_updates(sessions_home, tmp_path):
    """Same guarantee as the threaded test above, but with real OS
    processes -- the actual shape of the production race (a background
    worker process vs a Flask request process), not just threads sharing one
    interpreter."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src, session_id="mp-session")
    n = 12

    procs = [
        multiprocessing.Process(target=_mp_worker_record_edit, args=(sessions_home, sid, i))
        for i in range(n)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    entries = history(sid)
    assert len(entries) == n
    assert sorted(e["index"] for e in entries) == list(range(n))

    journal_file = sessions_home / sid / "history.json"
    raw = json.loads(journal_file.read_text(encoding="utf-8"))
    assert len(raw["stack"]) == n
    assert raw["pointer"] == n - 1


def test_concurrent_undo_and_record_edit_stay_consistent(sessions_home, tmp_path):
    """A mixed race -- some threads recording new edits while others undo --
    must never corrupt the journal, even though the two operations are
    fighting over the same pointer. The only hard guarantee this asserts is
    the one that matters: the journal stays valid JSON with a self-consistent
    pointer/stack, and record_edit's own return values were never silently
    dropped (each entry index handed back was actually written)."""
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    # Seed a few edits so there is something to undo concurrently with.
    for i in range(5):
        record_edit(sid, f"seed{i}", {}, str(tmp_path / f"seed{i}.mp3"))

    def _record(i):
        return record_edit(sid, f"op{i}", {}, str(tmp_path / f"out{i}.mp3"))["index"]

    def _undo(_i):
        undo(sid)
        return None

    tasks = [_record] * 15 + [_undo] * 15
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(lambda pair: pair[0](pair[1]), zip(tasks, range(len(tasks)))))

    recorded_indices = [r for r in results if r is not None]
    assert len(recorded_indices) == 15

    journal_file = sessions_home / sid / "history.json"
    raw = json.loads(journal_file.read_text(encoding="utf-8"))
    # Valid, self-consistent journal: pointer always points at a real slot
    # (or -1), and every recorded index actually made it into the stack.
    assert -1 <= raw["pointer"] < len(raw["stack"])
    stack_indices = {e["index"] for e in raw["stack"]}
    for idx in recorded_indices:
        assert idx in stack_indices


# ---------------------------------------------------------------------------
# prune_sessions
# ---------------------------------------------------------------------------


def test_prune_sessions_removes_stale_and_keeps_recent(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    old_sid = start_session(src, session_id="old-session")
    new_sid = start_session(src, session_id="new-session")

    # Backdate the old session's journal mtime past the retention window.
    old_journal = sessions_home / old_sid / "history.json"
    old_time = time.time() - 1000
    os.utime(old_journal, (old_time, old_time))

    removed = prune_sessions(max_age_seconds=500)
    assert removed == 1
    assert session_exists(old_sid) is False
    assert session_exists(new_sid) is True


def test_prune_sessions_never_touches_a_recently_touched_session(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    removed = prune_sessions(max_age_seconds=DEFAULT_SESSION_MAX_AGE_SECONDS)
    assert removed == 0
    assert session_exists(sid) is True


def test_prune_sessions_uses_default_max_age_when_none_given(sessions_home, tmp_path):
    src = str(tmp_path / "src.mp3")
    sid = start_session(src)
    # Default retention is a week -- a brand new session must survive it.
    assert prune_sessions(max_age_seconds=None) == 0
    assert session_exists(sid) is True


def test_prune_sessions_on_missing_root_is_a_noop(tmp_path, monkeypatch):
    missing_root = tmp_path / "does-not-exist"
    monkeypatch.setattr(session_mod, "_sessions_root", lambda: missing_root)
    assert prune_sessions(max_age_seconds=1) == 0
