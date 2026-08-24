"""Tests for the undo/redo edit-session journal."""

import json

import pytest

import praisonai_editor.session as session_mod
from praisonai_editor.session import (
    current_path,
    end_session,
    history,
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
