"""Tests for the YAML plan runner (praisonai_editor.yaml_plan) and its CLI
wiring (the ``apply`` subcommand), plus CLI-level coverage for the other
cli.py changes landed alongside it (``session jump``, ``demix``, widened
``normalize``/``convert`` flags).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import praisonai_editor.session as session_mod
from praisonai_editor.session import history, session_exists
from praisonai_editor.yaml_plan import PlanError, load_plan, run_plan


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_sine(tmp_path: Path, name: str = "sine.wav", duration: float = 6.0, freq: int = 440) -> str:
    ffmpeg = shutil.which("ffmpeg")
    out = tmp_path / name
    result = subprocess.run(
        [
            ffmpeg, "-y", "-nostdin",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}",
            "-ar", "48000", "-ac", "2",
            str(out),
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]
    return str(out)


@pytest.fixture
def sessions_home(tmp_path, monkeypatch):
    """Same convention as tests/test_session.py: isolate the on-disk session
    journal root so tests never touch the real ~/.praisonai/editor/sessions."""
    root = tmp_path / "sessions"
    monkeypatch.setattr(session_mod, "_sessions_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# load_plan validation
# ---------------------------------------------------------------------------


class TestLoadPlan:
    def test_missing_source_raises_plan_error(self):
        with pytest.raises(PlanError, match="source"):
            load_plan({"steps": [{"op": "master", "params": {}}]})

    def test_missing_steps_raises_plan_error(self):
        with pytest.raises(PlanError, match="steps"):
            load_plan({"source": "in.wav"})

    def test_empty_steps_raises_plan_error(self):
        with pytest.raises(PlanError, match="steps"):
            load_plan({"source": "in.wav", "steps": []})

    def test_unknown_op_raises_clear_plan_error_not_keyerror(self):
        with pytest.raises(PlanError, match="unknown op 'not_a_real_op'"):
            load_plan({"source": "in.wav", "steps": [{"op": "not_a_real_op", "params": {}}]})

    def test_step_missing_op_raises_plan_error(self):
        with pytest.raises(PlanError, match="op"):
            load_plan({"source": "in.wav", "steps": [{"params": {}}]})

    def test_concat_without_sources_raises_plan_error(self):
        with pytest.raises(PlanError, match="sources"):
            load_plan({"source": "in.wav", "steps": [{"op": "concat", "params": {}}]})

    def test_isolate_vocals_without_continue_with_raises_plan_error(self):
        with pytest.raises(PlanError, match="continue_with"):
            load_plan({"source": "in.wav", "steps": [{"op": "isolate_vocals", "params": {}}]})

    def test_plan_file_not_found_raises_plan_error_not_raw_oserror(self):
        with pytest.raises(PlanError, match="not found"):
            load_plan("/definitely/does/not/exist/plan.yaml")

    def test_valid_minimal_plan_loads(self):
        plan = load_plan(
            {"source": "in.wav", "steps": [{"op": "master", "params": {"preset": "speech"}}]}
        )
        assert plan["source"] == "in.wav"
        assert plan["steps"][0]["op"] == "master"

    def test_loads_from_real_yaml_file(self, tmp_path):
        yaml_text = """
source: in.wav
steps:
  - op: master
    params:
      preset: speech
output: out.m4a
"""
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(str(p))
        assert plan["source"] == "in.wav"
        assert plan["output"] == "out.m4a"


# ---------------------------------------------------------------------------
# run_plan: real multi-step ffmpeg pipeline + session recording
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestRunPlanEndToEnd:
    def test_remove_ranges_then_master_produces_real_output_and_session_history(
        self, tmp_path, sessions_home
    ):
        sine = _make_sine(tmp_path, duration=6.0)
        final_output = str(tmp_path / "final.m4a")

        plan = {
            "source": sine,
            "steps": [
                {"op": "remove_ranges", "params": {"ranges": ["1.0-2.0"]}},
                {"op": "master", "params": {"preset": "speech", "chain": []}},
            ],
            "output": final_output,
        }

        result = run_plan(plan)

        assert result["output_path"] == final_output
        assert Path(final_output).exists()
        assert len(result["steps"]) == 2
        assert result["steps"][0]["op"] == "remove_ranges"
        assert result["steps"][1]["op"] == "master"

        # A real session WAS created and recorded -- one entry per step.
        sid = result["session_id"]
        assert sid is not None
        assert session_exists(sid)
        entries = history(sid)
        assert len(entries) == 2
        assert [e["operation"] for e in entries] == ["remove_ranges", "master"]
        # The final recorded entry's path is the plan's declared output.
        assert entries[-1]["path"] == final_output

    def test_remove_ranges_accepts_yaml_list_ranges_not_just_strings(self, tmp_path, sessions_home):
        """Regression test for a real bug found by live (non-mocked) testing:
        YAML's own list syntax (`ranges: [[1.0, 1.5]]`) parses to a plain
        Python list, never a tuple -- but remove_time_ranges's
        parse_time_range() only recognizes an actual tuple as "(start, end)"
        and otherwise falls through to splitting str(spec) on a separator,
        which mangles a list's repr into a bogus timestamp (e.g. "[1.0, 1.5]"
        split on "," yields "[1.0" -- exactly the failure this test pins
        down). Every existing test in this file only used the STRING range
        format ("1.0-2.0"), so this path was never actually exercised."""
        sine = _make_sine(tmp_path, duration=5.0)
        output = str(tmp_path / "cut.wav")

        plan = {
            "source": sine,
            "steps": [{"op": "remove_ranges", "params": {"ranges": [[1.0, 1.5]]}}],
            "output": output,
        }

        result = run_plan(plan)

        assert result["output_path"] == output
        assert Path(output).exists()
        assert result["steps"][0]["removed_duration"] == pytest.approx(0.5, abs=0.05)

    def test_denoise_is_a_real_reachable_yaml_op(self, tmp_path, sessions_home):
        """Regression test: `denoise` existed at the package/CLI/Studio-API/
        Studio-UI layers but was missing from OP_NAMES, so a YAML plan using
        it raised "unknown op" -- the one layer it couldn't be used from.
        Real ffmpeg run, not mocked: proves the op is now actually wired
        end-to-end, not just accepted by validation."""
        sine = _make_sine(tmp_path, duration=4.0)
        output = str(tmp_path / "denoised.m4a")

        plan = {
            "source": sine,
            "steps": [{"op": "denoise", "params": {"noise_reduction": 20.0, "noise_floor": -40.0}}],
            "output": output,
        }

        result = run_plan(plan)

        assert result["output_path"] == output
        assert Path(output).exists()
        assert result["steps"][0]["op"] == "denoise"
        assert result["steps"][0]["artifacts"]["noise_reduction"] == "20.0"

        sid = result["session_id"]
        assert session_exists(sid)
        assert history(sid)[0]["operation"] == "denoise"

    def test_word_gaps_is_a_real_reachable_yaml_op(self, tmp_path, sessions_home):
        """word_gaps (praisonai_editor.word_gaps.shorten_word_gaps) reachable
        from a YAML plan via a `transcript_path` param -- the same idiom
        phrase_trim's own --transcript/transcript_path already uses, since
        run_plan only threads the current FILE between steps, never an
        in-memory TranscriptResult. Real ffmpeg run: a tone, 3s of real
        silence, another tone, concatenated -- proves the gap is actually
        cut, not just that the op is accepted by validation."""
        ffmpeg = shutil.which("ffmpeg")
        seg1 = tmp_path / "seg1.wav"
        gap = tmp_path / "gap.wav"
        seg2 = tmp_path / "seg2.wav"
        for path, args in (
            (seg1, ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]),
            (gap, ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "3"]),
            (seg2, ["-f", "lavfi", "-i", "sine=frequency=880:duration=1"]),
        ):
            r = subprocess.run(
                [ffmpeg, "-y", "-nostdin", *args, "-ar", "44100", "-ac", "1", str(path)],
                capture_output=True)
            assert r.returncode == 0, r.stderr.decode()[-800:]

        combined = tmp_path / "combined.wav"
        list_file = tmp_path / "list.txt"
        list_file.write_text(f"file '{seg1}'\nfile '{gap}'\nfile '{seg2}'\n")
        r = subprocess.run(
            [ffmpeg, "-y", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(combined)],
            capture_output=True)
        assert r.returncode == 0, r.stderr.decode()[-800:]

        transcript_path = tmp_path / "transcript.json"
        transcript_path.write_text(json.dumps({
            "text": "a b",
            "words": [{"text": "a", "start": 0.0, "end": 1.0, "confidence": 1.0},
                      {"text": "b", "start": 4.0, "end": 5.0, "confidence": 1.0}],
            "language": "en", "duration": 5.0,
        }))

        output = str(tmp_path / "shortened.wav")
        plan = {
            "source": str(combined),
            "steps": [{"op": "word_gaps", "params": {
                "transcript_path": str(transcript_path), "threshold": 1.0, "target": 0.4,
            }}],
            "output": output,
        }

        result = run_plan(plan)

        assert result["output_path"] == output
        assert Path(output).exists()
        assert result["steps"][0]["op"] == "word_gaps"
        assert result["steps"][0]["artifacts"]["gaps_shortened"] == "1"

        after = float(subprocess.run(
            [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", output],
            capture_output=True, text=True).stdout.strip())
        # Original 5.0s, 3.0s gap kept at 0.4s -> removed 2.6s -> ~2.4s.
        assert after == pytest.approx(5.0 - 2.6, abs=0.15)

    def test_explicit_session_id_is_created_when_not_yet_existing(self, tmp_path, sessions_home):
        sine = _make_sine(tmp_path, duration=3.0)
        plan = {
            "source": sine,
            "session": {"id": "my-explicit-session"},
            "steps": [{"op": "master", "params": {"preset": "speech", "chain": []}}],
            "output": str(tmp_path / "out.m4a"),
        }
        result = run_plan(plan)
        assert result["session_id"] == "my-explicit-session"
        assert session_exists("my-explicit-session")

    def test_resuming_an_existing_session_continues_its_history_without_reset(
        self, tmp_path, sessions_home
    ):
        sine = _make_sine(tmp_path, duration=3.0)
        first_plan = {
            "source": sine,
            "session": {"id": "resume-me"},
            "steps": [{"op": "master", "params": {"preset": "speech", "chain": []}}],
            "output": str(tmp_path / "out1.m4a"),
        }
        run_plan(first_plan)
        assert len(history("resume-me")) == 1

        second_plan = {
            "source": sine,  # ignored -- resumes from the existing session's current_path
            "session": {"id": "resume-me"},
            "steps": [{"op": "conform", "params": {"sample_rate": 44100}}],
            "output": str(tmp_path / "out2.m4a"),
        }
        result = run_plan(second_plan)
        assert result["session_id"] == "resume-me"
        entries = history("resume-me")
        # Two total entries now -- the second run APPENDED, it did not reset.
        assert len(entries) == 2
        assert entries[0]["operation"] == "master"
        assert entries[1]["operation"] == "conform"


# ---------------------------------------------------------------------------
# --no-session / session.record_history: false
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestRunPlanNoSession:
    def test_record_history_false_never_creates_a_journal(self, tmp_path, sessions_home):
        sine = _make_sine(tmp_path, duration=3.0)
        plan = {
            "source": sine,
            "session": {"record_history": False},
            "steps": [{"op": "master", "params": {"preset": "speech", "chain": []}}],
            "output": str(tmp_path / "out.m4a"),
        }
        result = run_plan(plan)
        assert result["session_id"] is None
        assert Path(result["output_path"]).exists()
        # No journal directory was ever created.
        assert not sessions_home.exists() or list(sessions_home.iterdir()) == []

    def test_no_session_flag_overrides_plan_default_and_never_creates_a_journal(
        self, tmp_path, sessions_home
    ):
        sine = _make_sine(tmp_path, duration=3.0)
        plan = {
            "source": sine,
            "steps": [{"op": "master", "params": {"preset": "speech", "chain": []}}],
            "output": str(tmp_path / "out.m4a"),
        }
        result = run_plan(plan, no_session=True)
        assert result["session_id"] is None
        assert Path(result["output_path"]).exists()
        assert not sessions_home.exists() or list(sessions_home.iterdir()) == []


# ---------------------------------------------------------------------------
# concat's explicit sources: exception to implicit chaining
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestRunPlanConcat:
    def test_concat_uses_explicit_sources_not_the_chained_current(self, tmp_path, sessions_home):
        sine_a = _make_sine(tmp_path, name="a.wav", duration=2.0, freq=440)
        sine_b = _make_sine(tmp_path, name="b.wav", duration=2.0, freq=220)
        joined = str(tmp_path / "joined.m4a")

        plan = {
            "source": sine_a,  # NOT one of concat's sources -- proves concat ignores "current"
            "steps": [
                {
                    "op": "concat",
                    "params": {"sources": [sine_a, sine_b], "reencode": True},
                },
            ],
            "output": joined,
        }
        result = run_plan(plan)
        assert result["output_path"] == joined
        assert Path(joined).exists()
        assert result["steps"][0]["sources"] == [sine_a, sine_b]


# ---------------------------------------------------------------------------
# CLI: apply
# ---------------------------------------------------------------------------


def _run_cli(args, env=None, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "praisonai_editor.cli", *args],
        capture_output=True, text=True, env=env, cwd=cwd,
    )


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestCliApply:
    def test_apply_runs_plan_end_to_end_via_subprocess(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        sine = _make_sine(tmp_path, duration=3.0)
        out_path = str(tmp_path / "final.m4a")
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            f"""
source: {sine}
steps:
  - op: master
    params:
      preset: speech
      chain: []
output: {out_path}
""",
            encoding="utf-8",
        )

        result = _run_cli(["apply", str(plan_path), "--json"], env=dict(os.environ))
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["output_path"] == out_path
        assert Path(out_path).exists()
        assert payload["session_id"] is not None

    def test_apply_unknown_op_reports_clean_error_not_traceback(self, tmp_path):
        import os

        plan_path = tmp_path / "bad_plan.yaml"
        plan_path.write_text(
            "source: in.wav\nsteps:\n  - op: totally_bogus_op\n    params: {}\n",
            encoding="utf-8",
        )
        result = _run_cli(["apply", str(plan_path)], env=dict(os.environ))
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "totally_bogus_op" in result.stderr

    def test_apply_no_session_flag_creates_no_journal(self, tmp_path):
        import os

        home = tmp_path / "home"
        home.mkdir()
        env = dict(os.environ)
        env["HOME"] = str(home)

        sine = _make_sine(tmp_path, duration=3.0)
        out_path = str(tmp_path / "final.m4a")
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            f"""
source: {sine}
steps:
  - op: master
    params:
      preset: speech
      chain: []
output: {out_path}
""",
            encoding="utf-8",
        )

        result = _run_cli(["apply", str(plan_path), "--no-session", "--json"], env=env)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["session_id"] is None
        sessions_root = home / ".praisonai" / "editor" / "sessions"
        assert not sessions_root.exists()


# ---------------------------------------------------------------------------
# CLI: session jump (real subprocess, negative index)
# ---------------------------------------------------------------------------


class TestCliSessionJump:
    def test_jump_negative_one_via_real_subprocess(self, tmp_path):
        import os

        home = tmp_path / "home"
        home.mkdir()
        env = dict(os.environ)
        env["HOME"] = str(home)

        start = _run_cli(["session", "start", "/tmp/fake_src.mp3", "--json"], env=env)
        assert start.returncode == 0, start.stderr
        sid = json.loads(start.stdout)["session_id"]

        # Record two edits by going through record_edit directly (in-process,
        # pointed at the same isolated HOME) -- record_edit itself has no
        # CLI subcommand; this just seeds history to jump through.
        sessions_root = home / ".praisonai" / "editor" / "sessions"
        assert sessions_root.is_dir()

        import praisonai_editor.session as sm
        sm._sessions_root = lambda: sessions_root
        sm.record_edit(sid, "op1", {}, "/tmp/out1.mp3")
        sm.record_edit(sid, "op2", {}, "/tmp/out2.mp3")

        jump_out = _run_cli(["session", "jump", sid, "-1", "--json"], env=env)
        assert jump_out.returncode == 0, jump_out.stderr
        payload = json.loads(jump_out.stdout)
        assert payload["path"] == "/tmp/fake_src.mp3"
        assert payload["index"] == -1

        jump_fwd = _run_cli(["session", "jump", sid, "1"], env=env)
        assert jump_fwd.returncode == 0, jump_fwd.stderr
        assert "out2.mp3" in jump_fwd.stdout

    def test_jump_unknown_session_reports_error(self, tmp_path):
        import os

        env = dict(os.environ)
        env["HOME"] = str(tmp_path / "home")
        result = _run_cli(["session", "jump", "no-such-session", "0"], env=env)
        assert result.returncode == 1
        assert "Unknown session" in result.stderr

    def test_jump_out_of_range_prints_nothing_to_jump_to(self, tmp_path):
        import os

        home = tmp_path / "home"
        home.mkdir()
        env = dict(os.environ)
        env["HOME"] = str(home)

        start = _run_cli(["session", "start", "/tmp/fake_src.mp3", "--json"], env=env)
        sid = json.loads(start.stdout)["session_id"]

        result = _run_cli(["session", "jump", sid, "5"], env=env)
        assert result.returncode == 0
        assert "Nothing to jump to." in result.stdout


# ---------------------------------------------------------------------------
# CLI: demix, normalize (new flags), convert (widened choices) -- arg wiring
# ---------------------------------------------------------------------------


class TestCliArgWiring:
    def test_normalize_new_loudness_flags_wired_through(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli
        import praisonai_editor.normalize as normalize_mod
        from praisonai_editor.normalize import NormalizeResult

        captured = {}

        def fake_optimize(input_path, output_path=None, **kwargs):
            captured["input_path"] = input_path
            captured["output_path"] = output_path
            captured.update(kwargs)
            return NormalizeResult(
                path=str(tmp_path / "out.m4a"), mean_db=-20.0, max_db=-2.0, normalized=True,
                target_lufs=kwargs.get("target_lufs", -16.0),
                true_peak_db=kwargs.get("true_peak_db", -1.5),
            )

        monkeypatch.setattr(normalize_mod, "optimize_audio_volume", fake_optimize)
        monkeypatch.setattr(cli, "cmd_normalize", cli.cmd_normalize)  # sanity: attr exists
        monkeypatch.setattr(
            sys, "argv",
            [
                "praisonai-editor", "normalize", "in.m4a",
                "-o", "out.m4a",
                "--target-lufs", "-18",
                "--true-peak", "-2.0",
                "--lra", "7",
            ],
        )
        assert cli.main() == 0
        assert captured["target_lufs"] == -18.0
        assert captured["true_peak_db"] == -2.0
        assert captured["lra"] == 7.0

    def test_convert_accepts_widened_format_choices(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli
        import praisonai_editor.convert as convert_mod

        for fmt in ("aac", "ogg", "flac"):
            captured = {}

            def fake_convert(input_path, output_path, **kwargs):
                captured["output_path"] = output_path
                return output_path

            monkeypatch.setattr(convert_mod, "convert_media", fake_convert)
            monkeypatch.setattr(
                sys, "argv", ["praisonai-editor", "convert", "in.wav", "--format", fmt],
            )
            assert cli.main() == 0
            assert captured["output_path"].endswith(f".{fmt}")

        # An unsupported format is still rejected by argparse, not silently accepted.
        monkeypatch.setattr(sys, "argv", ["praisonai-editor", "convert", "in.wav", "--format", "wma"])
        with pytest.raises(SystemExit):
            cli.main()

    def test_demix_cli_wiring_copies_stems_to_requested_outputs(self, monkeypatch, tmp_path):
        import praisonai_editor.cli as cli

        vocals_src = tmp_path / "cache_vocals.wav"
        inst_src = tmp_path / "cache_no_vocals.wav"
        vocals_src.write_bytes(b"VOCALS")
        inst_src.write_bytes(b"INSTRUMENTS")

        captured = {}

        def fake_isolate(media_path, **kwargs):
            captured["media_path"] = media_path
            captured.update(kwargs)
            return str(vocals_src), str(inst_src)

        import praisonai_editor._demix as demix_mod
        monkeypatch.setattr(demix_mod, "isolate_vocals", fake_isolate)

        vocals_out = str(tmp_path / "voc_out.wav")
        inst_out = str(tmp_path / "inst_out.wav")
        monkeypatch.setattr(
            sys, "argv",
            [
                "praisonai-editor", "demix", "in.mp3",
                "--vocals-output", vocals_out,
                "--instruments-output", inst_out,
                "--model", "mdx_extra_q",
                "--device", "cpu",
                "--json",
            ],
        )
        assert cli.main() == 0
        assert captured["model_name"] == "mdx_extra_q"
        assert captured["device"] == "cpu"
        assert Path(vocals_out).read_bytes() == b"VOCALS"
        assert Path(inst_out).read_bytes() == b"INSTRUMENTS"
