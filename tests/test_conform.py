"""Tests for audio conforming (pure logic — no ffmpeg execution)."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import praisonai_editor.conform as conform_mod
from praisonai_editor.conform import _build_conform_filter, conform_audio


@pytest.fixture
def fake_run(monkeypatch):
    """Monkeypatch subprocess.run to capture the ffmpeg command."""
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(conform_mod, "_find_ffmpeg", lambda: "ffmpeg")
    return calls


def _make_input(tmp_path, name="src.wav"):
    p = tmp_path / name
    p.write_bytes(b"\x00")
    return str(p)


class TestValidation:
    def test_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            conform_audio(str(tmp_path / "nope.wav"))

    def test_bad_channels(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError):
            conform_audio(src, channels=3)

    def test_zero_duration(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError):
            conform_audio(src, duration=0)

    def test_negative_duration(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError):
            conform_audio(src, duration=-5.0)


class TestFilterConstruction:
    def test_stereo_no_duration(self):
        af = _build_conform_filter(48000, 2, None)
        assert af == "aformat=sample_rates=48000:channel_layouts=stereo"

    def test_mono(self):
        af = _build_conform_filter(44100, 1, None)
        assert af == "aformat=sample_rates=44100:channel_layouts=mono"

    def test_with_duration(self):
        af = _build_conform_filter(48000, 2, 12.5)
        assert af == (
            "aformat=sample_rates=48000:channel_layouts=stereo,"
            "atrim=0:12.5,apad=whole_dur=12.5"
        )


class TestOutputPath:
    def test_default_output_path(self, tmp_path, fake_run):
        src = _make_input(tmp_path, "mastered.wav")
        result = conform_audio(src)
        assert result == str(tmp_path / "mastered_conformed.m4a")
        assert fake_run[0][-1] == result

    def test_explicit_output_path(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        out = str(tmp_path / "custom.m4a")
        result = conform_audio(src, out)
        assert result == out


class TestCommandConstruction:
    def test_command_shape(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        out = str(tmp_path / "out.m4a")
        conform_audio(src, out, bitrate="128k")

        cmd = fake_run[0]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd and "-nostdin" in cmd
        assert cmd[cmd.index("-i") + 1] == src
        af = cmd[cmd.index("-af") + 1]
        assert af == "aformat=sample_rates=48000:channel_layouts=stereo"
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[cmd.index("-b:a") + 1] == "128k"
        assert cmd[-1] == out

    def test_command_with_duration_and_mono(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        conform_audio(src, sample_rate=44100, channels=1, duration=30.0)

        cmd = fake_run[0]
        af = cmd[cmd.index("-af") + 1]
        assert af == (
            "aformat=sample_rates=44100:channel_layouts=mono,"
            "atrim=0:30.0,apad=whole_dur=30.0"
        )


class TestFailure:
    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        src = _make_input(tmp_path)
        monkeypatch.setattr(conform_mod, "_find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            conform_audio(src)
