"""Tests for audio concatenation (pure logic — no ffmpeg execution)."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import praisonai_editor.concat as concat_mod
from praisonai_editor.concat import concat_audio


@pytest.fixture
def fake_run(monkeypatch):
    """Monkeypatch subprocess.run to capture the ffmpeg command (and list file)."""
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        record = {"cmd": list(cmd)}
        # Concat demuxer: read the list file NOW — the tempdir is gone afterwards.
        if "-f" in cmd and "concat" in cmd:
            list_path = Path(cmd[cmd.index("-i") + 1])
            record["list_content"] = list_path.read_text(encoding="utf-8")
        calls.append(record)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(concat_mod, "_find_ffmpeg", lambda: "ffmpeg")
    return calls


def _make_inputs(tmp_path, names):
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"\x00")
        paths.append(str(p))
    return paths


class TestValidation:
    def test_empty_inputs(self, tmp_path):
        with pytest.raises(ValueError):
            concat_audio([], str(tmp_path / "out.m4a"))

    def test_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            concat_audio([str(tmp_path / "nope.m4a")], str(tmp_path / "out.m4a"))

    def test_one_missing_among_existing(self, tmp_path):
        (a,) = _make_inputs(tmp_path, ["a.m4a"])
        with pytest.raises(FileNotFoundError):
            concat_audio([a, str(tmp_path / "missing.m4a")], str(tmp_path / "out.m4a"))


class TestCopyPath:
    def test_command_uses_concat_demuxer_and_copy(self, tmp_path, fake_run):
        inputs = _make_inputs(tmp_path, ["a.m4a", "b.m4a"])
        out = str(tmp_path / "out.m4a")

        result = concat_audio(inputs, out)

        assert result == out
        cmd = fake_run[0]["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd and "-nostdin" in cmd
        assert cmd[cmd.index("-f") + 1] == "concat"
        assert cmd[cmd.index("-safe") + 1] == "0"
        assert cmd[cmd.index("-c") + 1] == "copy"
        assert cmd[-1] == out

    def test_list_file_contains_absolute_paths(self, tmp_path, fake_run):
        inputs = _make_inputs(tmp_path, ["a.m4a", "b.m4a"])
        concat_audio(inputs, str(tmp_path / "out.m4a"))

        content = fake_run[0]["list_content"]
        lines = content.strip().splitlines()
        assert len(lines) == 2
        for line, src in zip(lines, inputs):
            assert line == f"file '{Path(src).resolve()}'"

    def test_list_file_escapes_single_quotes(self, tmp_path, fake_run):
        inputs = _make_inputs(tmp_path, ["it's here.m4a"])
        concat_audio(inputs, str(tmp_path / "out.m4a"))

        content = fake_run[0]["list_content"]
        expected = str(Path(inputs[0]).resolve()).replace("'", "'\\''")
        assert f"file '{expected}'" in content


class TestReencodePath:
    def test_command_uses_filter_complex(self, tmp_path, fake_run):
        inputs = _make_inputs(tmp_path, ["a.m4a", "b.mp3", "c.wav"])
        out = str(tmp_path / "out.m4a")

        result = concat_audio(inputs, out, reencode=True, bitrate="128k")

        assert result == out
        cmd = fake_run[0]["cmd"]
        assert cmd.count("-i") == 3
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "concat=n=3:v=0:a=1[out]" in fc
        assert fc.count("aformat=sample_rates=48000:channel_layouts=stereo") == 3
        assert "[0:a]" in fc and "[1:a]" in fc and "[2:a]" in fc
        assert cmd[cmd.index("-map") + 1] == "[out]"
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[cmd.index("-b:a") + 1] == "128k"
        assert cmd[-1] == out

    def test_no_list_file_when_reencoding(self, tmp_path, fake_run):
        inputs = _make_inputs(tmp_path, ["a.m4a", "b.m4a"])
        concat_audio(inputs, str(tmp_path / "out.m4a"), reencode=True)

        cmd = fake_run[0]["cmd"]
        assert "-f" not in cmd
        assert "copy" not in cmd


class TestFailure:
    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        inputs = _make_inputs(tmp_path, ["a.m4a"])
        monkeypatch.setattr(concat_mod, "_find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            concat_audio(inputs, str(tmp_path / "out.m4a"))
