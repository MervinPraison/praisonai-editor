"""Tests for FFT-based noise reduction (afftdn)."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import praisonai_editor.denoise as denoise_mod
from praisonai_editor.denoise import _build_denoise_filter, denoise_audio
from praisonai_editor.models import EditResult


def _make_input(tmp_path, name="src.wav"):
    p = tmp_path / name
    p.write_bytes(b"\x00")
    return str(p)


@pytest.fixture
def fake_run(monkeypatch):
    """Monkeypatch subprocess.run to capture the ffmpeg command."""
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(denoise_mod, "_find_ffmpeg", lambda: "ffmpeg")
    return calls


class TestFilterConstruction:
    def test_default_filter(self):
        af = _build_denoise_filter(12.0, -50.0, True)
        assert af == "afftdn=nr=12:nf=-50:tn=1"

    def test_track_noise_off(self):
        af = _build_denoise_filter(12.0, -50.0, False)
        assert af == "afftdn=nr=12:nf=-50:tn=0"

    def test_custom_values(self):
        af = _build_denoise_filter(30.5, -35.0, True)
        assert af == "afftdn=nr=30.5:nf=-35:tn=1"


class TestValidation:
    def test_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            denoise_audio(str(tmp_path / "nope.wav"))

    def test_noise_reduction_too_low(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError, match="noise_reduction"):
            denoise_audio(src, noise_reduction=0.0)

    def test_noise_reduction_too_high(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError, match="noise_reduction"):
            denoise_audio(src, noise_reduction=98.0)

    def test_noise_floor_too_low(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError, match="noise_floor"):
            denoise_audio(src, noise_floor=-81.0)

    def test_noise_floor_too_high(self, tmp_path):
        src = _make_input(tmp_path)
        with pytest.raises(ValueError, match="noise_floor"):
            denoise_audio(src, noise_floor=-19.0)

    def test_boundary_values_are_valid(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        denoise_audio(src, noise_reduction=0.01, noise_floor=-80.0)
        denoise_audio(src, noise_reduction=97.0, noise_floor=-20.0)


class TestOutputPath:
    def test_default_output_path(self, tmp_path, fake_run):
        src = _make_input(tmp_path, "src.wav")
        result = denoise_audio(src)
        assert result.output_path == str(tmp_path / "src_denoised.m4a")
        assert fake_run[0][-1] == result.output_path

    def test_explicit_output_path(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        out = str(tmp_path / "custom.m4a")
        result = denoise_audio(src, out)
        assert result.output_path == out


class TestCommandConstruction:
    def test_command_shape(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        out = str(tmp_path / "out.wav")
        denoise_audio(src, out, noise_reduction=20.0, noise_floor=-40.0, bitrate="128k")

        cmd = fake_run[0]
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd and "-nostdin" in cmd
        assert cmd[cmd.index("-i") + 1] == src
        af = cmd[cmd.index("-af") + 1]
        assert af == "afftdn=nr=20:nf=-40:tn=1"
        assert cmd[cmd.index("-c:a") + 1] == "aac"
        assert cmd[cmd.index("-b:a") + 1] == "128k"
        assert cmd[-1] == out

    def test_track_noise_disabled(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        denoise_audio(src, track_noise=False)
        cmd = fake_run[0]
        af = cmd[cmd.index("-af") + 1]
        assert af.endswith("tn=0")

    def test_returns_edit_result(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        result = denoise_audio(src)
        assert isinstance(result, EditResult)
        assert result.input_path == src
        assert result.success is True


class TestFailure:
    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        src = _make_input(tmp_path)
        monkeypatch.setattr(denoise_mod, "_find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            denoise_audio(src)


class TestCli:
    def test_denoise_arg_wiring(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli

        captured = {}

        def fake_denoise_audio(input_path, output_path=None, **kwargs):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured.update(kwargs)
            return EditResult(
                input_path=input_path,
                output_path=str(tmp_path / "out.m4a"),
                success=True,
            )

        monkeypatch.setattr(denoise_mod, "denoise_audio", fake_denoise_audio)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "praisonai-editor", "denoise", "in.m4a",
                "-o", "out.m4a",
                "--noise-reduction", "25",
                "--noise-floor", "-35",
                "--no-track-noise",
                "--bitrate", "128k",
                "--json",
            ],
        )
        assert cli.main() == 0

        assert captured == {
            "input_path": "in.m4a",
            "output_path": "out.m4a",
            "noise_reduction": 25.0,
            "noise_floor": -35.0,
            "track_noise": False,
            "bitrate": "128k",
            "verbose": False,
        }

    def test_denoise_defaults(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli

        captured = {}

        def fake_denoise_audio(input_path, output_path=None, **kwargs):
            captured["output_path"] = output_path
            captured.update(kwargs)
            return EditResult(
                input_path=input_path,
                output_path=str(tmp_path / "in_denoised.m4a"),
                success=True,
            )

        monkeypatch.setattr(denoise_mod, "denoise_audio", fake_denoise_audio)
        monkeypatch.setattr(sys, "argv", ["praisonai-editor", "denoise", "in.m4a"])
        assert cli.main() == 0
        assert captured["output_path"] is None
        assert captured["noise_reduction"] == 12.0
        assert captured["noise_floor"] == -50.0
        assert captured["track_noise"] is True
        assert captured["bitrate"] == "192k"


def _ffmpeg_available() -> bool:
    try:
        denoise_mod._find_ffmpeg()
        return True
    except FileNotFoundError:
        return False


def _mean_volume_db(path: str) -> float:
    """Measure mean_volume (dB) of a file via ffmpeg's own volumedetect filter."""
    import re

    ffmpeg = denoise_mod._find_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]
    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", result.stderr)
    assert m, f"volumedetect parse failed: {result.stderr[-800:]}"
    return float(m.group(1))


def _gen_noise(ffmpeg: str, path: Path, amplitude: float, duration: float = 3.0) -> None:
    gen = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-f", "lavfi",
         "-i", f"anoisesrc=color=white:amplitude={amplitude}:duration={duration}",
         "-ar", "48000", "-ac", "2", str(path)],
        capture_output=True,
    )
    assert gen.returncode == 0, gen.stderr.decode()[-800:]


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_integration_default_settings_reduce_noise(tmp_path):
    """Real ffmpeg, default params: build a background-hiss clip (white
    noise, no other signal — the "silent-in-the-original-but-noisy" case),
    denoise it with every parameter left at its default, and prove via
    volumedetect that its measured level drops after denoising — i.e. real
    noise reduction happened with the out-of-the-box settings, not just a
    re-encode.

    (A tone mixed WITH noise is a poor discriminator for this assertion:
    afftdn is designed to preserve tonal/harmonic content while suppressing
    the broadband noise around it, so the tone continues to dominate
    mean_volume regardless of how much noise was actually removed. Pure
    background noise isolates the metric to what this feature controls.

    The default ``track_noise=True`` mode adapts gradually and is
    deliberately conservative on perfectly uniform noise with no quiet
    reference to calibrate against, so the reduction here is real but
    modest — see the tuned-settings test below for a dramatic reduction.)
    """
    ffmpeg = denoise_mod._find_ffmpeg()
    noisy = tmp_path / "noisy.wav"
    _gen_noise(ffmpeg, noisy, amplitude=0.05)

    before_mean = _mean_volume_db(str(noisy))

    result = denoise_audio(str(noisy), str(tmp_path / "denoised.m4a"))
    assert Path(result.output_path).exists()
    assert result.success is True

    after_mean = _mean_volume_db(result.output_path)
    assert after_mean < before_mean - 1.0


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_integration_tuned_settings_dramatically_reduce_noise(tmp_path):
    """Real ffmpeg, explicit params: with ``noise_floor`` set close to the
    clip's actual noise level and ``track_noise=False`` (a static profile,
    no adaptive learning curve to wait out), afftdn should suppress a quiet
    background-hiss clip almost entirely — a large, unambiguous reduction
    that proves the underlying filter is doing real, effective work.
    """
    ffmpeg = denoise_mod._find_ffmpeg()
    noisy = tmp_path / "noisy_quiet.wav"
    _gen_noise(ffmpeg, noisy, amplitude=0.01)

    before_mean = _mean_volume_db(str(noisy))

    result = denoise_audio(
        str(noisy),
        str(tmp_path / "denoised_quiet.m4a"),
        noise_reduction=97.0,
        noise_floor=-20.0,
        track_noise=False,
    )
    assert Path(result.output_path).exists()

    after_mean = _mean_volume_db(result.output_path)
    assert after_mean < before_mean - 15.0


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_integration_default_output_naming(tmp_path):
    """Real ffmpeg: default output path is {stem}_denoised.m4a next to the input."""
    ffmpeg = denoise_mod._find_ffmpeg()
    src = tmp_path / "clip.wav"
    gen = subprocess.run(
        [ffmpeg, "-y", "-nostdin", "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
         "-ar", "48000", "-ac", "1", str(src)],
        capture_output=True,
    )
    assert gen.returncode == 0, gen.stderr.decode()[-800:]

    result = denoise_audio(str(src))
    assert result.output_path == str(tmp_path / "clip_denoised.m4a")
    assert Path(result.output_path).exists()
