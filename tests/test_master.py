"""Tests for two-pass loudness mastering (pure logic + one ffmpeg integration test)."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import praisonai_editor.master as master_mod
from praisonai_editor.master import (
    MASTER_PRESETS,
    LoudnessStats,
    MasterResult,
    _build_master_filter,
    _parse_loudnorm_json,
    _pick_preset,
    master_audio,
    measure_loudness,
)

# Realistic loudnorm stderr: banner noise + values as JSON strings.
CANNED_STDERR = """\
size=N/A time=00:59:00.10 bitrate=N/A speed= 196x
[Parsed_loudnorm_0 @ 0x600000c04d20]
{
\t"input_i" : "-23.62",
\t"input_tp" : "-6.47",
\t"input_lra" : "18.06",
\t"input_thresh" : "-34.01",
\t"output_i" : "-14.46",
\t"output_tp" : "-2.10",
\t"output_lra" : "16.00",
\t"output_thresh" : "-24.87",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.46"
}
"""

SILENT_STDERR = """\
[Parsed_loudnorm_0 @ 0x600000c04d20]
{
\t"input_i" : "-inf",
\t"input_tp" : "-inf",
\t"input_lra" : "0.00",
\t"input_thresh" : "-inf",
\t"output_i" : "-14.00",
\t"output_tp" : "-1.50",
\t"output_lra" : "0.00",
\t"output_thresh" : "-24.00",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.00"
}
"""

SPEECH_STDERR = CANNED_STDERR.replace('"-23.62"', '"-19.10"').replace('"18.06"', '"6.20"')


def _make_input(tmp_path, name="src.m4a"):
    p = tmp_path / name
    p.write_bytes(b"\x00")
    return str(p)


def _fake_run_factory(calls, measure_stderr):
    """subprocess.run stand-in: canned stderr for the '-f null' measure pass."""

    def _fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if "null" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr=measure_stderr)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return _fake_run


@pytest.fixture
def fake_run(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls, CANNED_STDERR))
    monkeypatch.setattr(master_mod, "_find_ffmpeg", lambda: "ffmpeg")
    return calls


class TestParseLoudnormJson:
    def test_parses_string_values(self):
        stats = _parse_loudnorm_json(CANNED_STDERR)
        assert stats == LoudnessStats(
            input_i=-23.62,
            input_tp=-6.47,
            input_lra=18.06,
            input_thresh=-34.01,
            target_offset=0.46,
        )

    def test_parses_inf_values(self):
        stats = _parse_loudnorm_json(SILENT_STDERR)
        assert stats.input_i == float("-inf")
        assert stats.input_tp == float("-inf")
        assert stats.input_lra == 0.0

    def test_takes_last_json_block(self):
        earlier = CANNED_STDERR.replace('"-23.62"', '"-50.00"')
        stats = _parse_loudnorm_json(earlier + "\nmore noise\n" + CANNED_STDERR)
        assert stats.input_i == -23.62

    def test_parse_failure_raises(self):
        with pytest.raises(RuntimeError, match="parse failed"):
            _parse_loudnorm_json("no json here {not: valid}")


class TestMeasureLoudness:
    def test_measure_command_and_result(self, fake_run):
        stats = measure_loudness("in.m4a")
        assert stats.input_i == -23.62
        cmd = fake_run[0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-i") + 1] == "in.m4a"
        assert cmd[cmd.index("-af") + 1] == "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json"
        assert cmd[-2:] == ["null", "-"]

    def test_measure_failure_raises(self, monkeypatch):
        monkeypatch.setattr(master_mod, "_find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            measure_loudness("in.m4a")


class TestPresets:
    def test_preset_values(self):
        assert MASTER_PRESETS["speech"]["lra"] == 11.0
        assert MASTER_PRESETS["music"]["lra"] == 15.0
        assert MASTER_PRESETS["speech"]["pre_chain"] == [
            "acompressor=threshold=-18dB:ratio=3:attack=20:release=250:makeup=4dB"
        ]
        assert MASTER_PRESETS["music"]["pre_chain"] == [
            "acompressor=threshold=-16dB:ratio=2:attack=25:release=300:makeup=2dB"
        ]

    def test_auto_heuristic_wide_lra_is_music(self):
        stats = LoudnessStats(-20.0, -3.0, 18.0, -30.0, 0.1)
        assert _pick_preset(stats) == "music"

    def test_auto_heuristic_narrow_lra_is_speech(self):
        stats = LoudnessStats(-20.0, -3.0, 6.0, -30.0, 0.1)
        assert _pick_preset(stats) == "speech"


class TestBuildMasterFilter:
    def test_full_chain(self):
        stats = LoudnessStats(-23.62, -6.47, 18.06, -34.01, 0.46)
        af = _build_master_filter(
            MASTER_PRESETS["speech"]["pre_chain"], stats, -14.0, -1.5, 11.0, 48000
        )
        assert af == (
            "acompressor=threshold=-18dB:ratio=3:attack=20:release=250:makeup=4dB,"
            "loudnorm=I=-14:TP=-1.5:LRA=11"
            ":measured_I=-23.62:measured_TP=-6.47:measured_LRA=18.06"
            ":measured_thresh=-34.01:offset=0.46:linear=true,"
            "alimiter=limit=0.891:level=false,"
            "aresample=48000"
        )


class TestMasterAudio:
    def test_validation(self, tmp_path, fake_run):
        with pytest.raises(FileNotFoundError):
            master_audio(str(tmp_path / "nope.m4a"))
        src = _make_input(tmp_path)
        with pytest.raises(ValueError, match="preset"):
            master_audio(src, preset="edm")
        with pytest.raises(ValueError, match="channels"):
            master_audio(src, channels=3)

    def test_default_output_and_command_shape(self, tmp_path, fake_run):
        src = _make_input(tmp_path, "talk.m4a")
        result = master_audio(src)
        assert result.path == str(tmp_path / "talk.mastered.m4a")
        assert result.preset == "speech"
        assert result.normalized is True
        assert result.stats.input_i == -23.62

        measure_cmd, encode_cmd = fake_run
        assert "null" in measure_cmd
        assert encode_cmd[0] == "ffmpeg"
        assert "-y" in encode_cmd and "-nostdin" in encode_cmd
        assert encode_cmd[encode_cmd.index("-i") + 1] == src
        assert encode_cmd[encode_cmd.index("-ac") + 1] == "2"
        assert encode_cmd[encode_cmd.index("-c:a") + 1] == "aac"
        assert encode_cmd[encode_cmd.index("-b:a") + 1] == "192k"
        assert encode_cmd[-1] == result.path

        af = encode_cmd[encode_cmd.index("-af") + 1]
        assert af == result.chain
        assert af.startswith(MASTER_PRESETS["speech"]["pre_chain"][0] + ",")
        assert "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-23.62" in af
        assert "linear=true" in af
        assert af.endswith("alimiter=limit=0.891:level=false,aresample=48000")

    def test_music_preset_lra_and_options(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        result = master_audio(
            src,
            preset="music",
            target_lufs=-16.0,
            true_peak_db=-2.0,
            sample_rate=44100,
            channels=1,
            bitrate="128k",
        )
        encode_cmd = fake_run[1]
        af = encode_cmd[encode_cmd.index("-af") + 1]
        assert af.startswith(MASTER_PRESETS["music"]["pre_chain"][0] + ",")
        assert "loudnorm=I=-16:TP=-2:LRA=15:" in af
        assert af.endswith("aresample=44100")
        assert encode_cmd[encode_cmd.index("-ac") + 1] == "1"
        assert encode_cmd[encode_cmd.index("-b:a") + 1] == "128k"
        assert result.preset == "music"

    def test_lra_override(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        master_audio(src, lra=9.0)
        encode_cmd = fake_run[1]
        assert ":LRA=9:" in encode_cmd[encode_cmd.index("-af") + 1]

    def test_chain_overrides_preset_pre_chain(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        result = master_audio(src, chain=["highpass=f=80", "afftdn=nr=12:nf=-25"])
        af = fake_run[1][fake_run[1].index("-af") + 1]
        assert af.startswith("highpass=f=80,afftdn=nr=12:nf=-25,loudnorm=")
        assert "acompressor" not in af
        assert result.chain == af

    def test_empty_chain_is_loudnorm_only(self, tmp_path, fake_run):
        src = _make_input(tmp_path)
        master_audio(src, chain=[])
        af = fake_run[1][fake_run[1].index("-af") + 1]
        assert af.startswith("loudnorm=")
        assert "acompressor" not in af

    def test_auto_preset_resolves_from_stats(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls, CANNED_STDERR))
        monkeypatch.setattr(master_mod, "_find_ffmpeg", lambda: "ffmpeg")
        src = _make_input(tmp_path)
        result = master_audio(src, preset="auto")  # canned LRA 18.06 → music
        assert result.preset == "music"
        af = calls[1][calls[1].index("-af") + 1]
        assert af.startswith(MASTER_PRESETS["music"]["pre_chain"][0] + ",")

        calls.clear()
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls, SPEECH_STDERR))
        result = master_audio(src, preset="auto")  # narrow LRA 6.2 → speech
        assert result.preset == "speech"

    def test_silence_skips_loudnorm(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls, SILENT_STDERR))
        monkeypatch.setattr(master_mod, "_find_ffmpeg", lambda: "ffmpeg")
        src = _make_input(tmp_path)
        result = master_audio(src)
        assert result.normalized is False
        assert result.chain == "aresample=48000"
        af = calls[1][calls[1].index("-af") + 1]
        assert "loudnorm" not in af and "acompressor" not in af

    def test_encode_failure_raises(self, tmp_path, monkeypatch):
        def _fail_encode(cmd, *args, **kwargs):
            if "null" in cmd:
                return SimpleNamespace(returncode=0, stdout="", stderr=CANNED_STDERR)
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"encode boom")

        monkeypatch.setattr(subprocess, "run", _fail_encode)
        monkeypatch.setattr(master_mod, "_find_ffmpeg", lambda: "ffmpeg")
        with pytest.raises(RuntimeError, match="encode boom"):
            master_audio(_make_input(tmp_path))


class TestCli:
    def test_master_arg_wiring(self, monkeypatch, tmp_path, capsys):
        import praisonai_editor.cli as cli

        captured = {}

        def fake_master_audio(input_path, output_path=None, **kwargs):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured.update(kwargs)
            return MasterResult(
                path=str(tmp_path / "out.m4a"),
                stats=LoudnessStats(-23.62, -6.47, 18.06, -34.01, 0.46),
                preset="music",
                chain="af-chain",
                target_lufs=-16.0,
                true_peak_db=-2.0,
                normalized=True,
            )

        monkeypatch.setattr(master_mod, "master_audio", fake_master_audio)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "praisonai-editor", "master", "in.m4a",
                "-o", "out.m4a",
                "--preset", "music",
                "--target-lufs", "-16",
                "--true-peak", "-2",
                "--lra", "9",
                "--sample-rate", "44100",
                "--channels", "1",
                "--bitrate", "128k",
                "--json",
            ],
        )
        assert cli.main() == 0

        assert captured == {
            "input_path": "in.m4a",
            "output_path": "out.m4a",
            "preset": "music",
            "target_lufs": -16.0,
            "true_peak_db": -2.0,
            "lra": 9.0,
            "sample_rate": 44100,
            "channels": 1,
            "bitrate": "128k",
            "verbose": False,
        }

        payload = json.loads(capsys.readouterr().out)
        assert payload["preset"] == "music"
        assert payload["normalized"] is True
        assert payload["stats"]["input_i"] == -23.62

    def test_master_defaults(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli

        captured = {}

        def fake_master_audio(input_path, output_path=None, **kwargs):
            captured["output_path"] = output_path
            captured.update(kwargs)
            return MasterResult(
                path=str(tmp_path / "in.mastered.m4a"),
                stats=LoudnessStats(-23.62, -6.47, 18.06, -34.01, 0.46),
                preset="speech",
                chain="af-chain",
                target_lufs=-14.0,
                true_peak_db=-1.5,
            )

        monkeypatch.setattr(master_mod, "master_audio", fake_master_audio)
        monkeypatch.setattr(sys, "argv", ["praisonai-editor", "master", "in.m4a"])
        assert cli.main() == 0
        assert captured["output_path"] is None
        assert captured["preset"] == "speech"
        assert captured["target_lufs"] == -14.0
        assert captured["true_peak_db"] == -1.5
        assert captured["lra"] is None
        assert captured["sample_rate"] == 48000
        assert captured["channels"] == 2
        assert captured["bitrate"] == "192k"


def _ffmpeg_available() -> bool:
    try:
        master_mod._find_ffmpeg()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_integration_sine_masters_to_minus_14_lufs(tmp_path):
    """Real ffmpeg: master a 3 s sine and verify integrated loudness lands at -14 LUFS."""
    ffmpeg = master_mod._find_ffmpeg()
    sine = tmp_path / "sine.wav"
    gen = subprocess.run(
        [
            ffmpeg, "-y", "-nostdin",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=3",
            "-ar", "48000",
            "-ac", "2",
            str(sine),
        ],
        capture_output=True,
    )
    assert gen.returncode == 0, gen.stderr.decode()[-800:]

    # chain=[] → pure two-pass loudnorm (a compressor ahead of loudnorm would
    # invalidate the pass-1 measurement of a full-scale test tone).
    result = master_audio(str(sine), str(tmp_path / "mastered.m4a"), chain=[])
    assert Path(result.path).exists()
    assert result.normalized is True

    stats = measure_loudness(result.path)
    assert abs(stats.input_i - (-14.0)) <= 1.0
