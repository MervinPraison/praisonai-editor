"""Regression tests for pipeline.py's edit_media/edit_video on VIDEO input.

Found by a parity/gap audit: edit_media() unconditionally forwards demix= and
primary_zone_only= to edit_video(), but edit_video()'s signature had no such
parameters (and no **kwargs) -- every call to edit_media()/CLI `edit`/YAML
`edit`/`preset_edit` against a video file raised TypeError immediately,
regardless of preset or flag values. A second, related bug: edit_video()'s
content-detection branch unpacked create_content_plan()'s return as a 2-tuple
(`plan, blocks = ...`) when the function actually returns a 3-tuple
(EditPlan, List[ContentBlock], List[ContentBlock]) -- a ValueError on every
songs_only/speech_only/no_silence preset for video, independent of the first
bug. No test anywhere referenced edit_video/edit_media before this file.

Real ffmpeg-built video fixture, real FFmpegProber/FFmpegVideoRenderer --
only the network-calling transcriber is mocked (same convention as this
repo's other tests that avoid requiring an OpenAI API key for pure
pipeline-wiring coverage), and artifacts_dir is real (Path.home()-based, no
override param exists) but scoped to a throwaway, cleaned-up stem name so
nothing leaks into a real user's ~/.praisonai/editor.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from praisonai_editor.models import TranscriptResult, Word
from praisonai_editor.pipeline import edit_media, edit_video


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_test_video(path: Path, duration: float = 3.0) -> None:
    ffmpeg = shutil.which("ffmpeg")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-nostdin",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=15",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()[-800:]


@pytest.fixture(autouse=True)
def _isolate_artifacts_dir(monkeypatch, tmp_path):
    """edit_video/edit_audio hardcode Path.home() / '.praisonai/editor/<stem>'
    with no override param -- redirect Path.home() for the duration of the
    test so nothing is created under the real user's home directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def _fake_transcript() -> TranscriptResult:
    words = [
        Word(text="hello", start=0.0, end=0.4, confidence=0.99),
        Word(text="world", start=0.5, end=0.9, confidence=0.99),
    ]
    return TranscriptResult(text="hello world", words=words, language="en", duration=3.0)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestEditVideoAcceptsDemixKwargs:
    """The exact TypeError edit_media() raised on every video input, for
    every preset, regardless of demix/primary_zone_only's values."""

    def test_edit_media_on_video_does_not_raise_typeerror(self, tmp_path, monkeypatch):
        video = tmp_path / "clip.mp4"
        _make_test_video(video)
        monkeypatch.setattr(
            "praisonai_editor.pipeline.OpenAITranscriber",
            lambda: type("T", (), {"transcribe": staticmethod(lambda *a, **k: _fake_transcript())})(),
        )

        result = edit_media(
            str(video), output_path=str(tmp_path / "out.mp4"),
            preset="podcast", save_artifacts=False,
        )

        assert result.success, result.error
        assert Path(result.output_path).exists()

    def test_edit_video_directly_accepts_demix_and_primary_zone_only(self, tmp_path, monkeypatch):
        video = tmp_path / "clip2.mp4"
        _make_test_video(video)
        monkeypatch.setattr(
            "praisonai_editor.pipeline.OpenAITranscriber",
            lambda: type("T", (), {"transcribe": staticmethod(lambda *a, **k: _fake_transcript())})(),
        )

        result = edit_video(
            str(video), output_path=str(tmp_path / "out2.mp4"),
            preset="podcast", demix=True, primary_zone_only=True,
            save_artifacts=False,
        )

        assert result.success, result.error


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
class TestEditVideoContentPresetUnpacksAllThreeReturnValues:
    """create_content_plan() always returns a 3-tuple (plan, blocks,
    all_events) -- edit_video's content-detection branch used to unpack only
    2, raising ValueError on every songs_only/speech_only/no_silence video
    edit, independent of the demix kwarg bug above."""

    def test_no_silence_preset_with_demix_and_primary_zone_only(self, tmp_path, monkeypatch):
        """no_silence's keep_types spans every non-silence content type, so
        this doesn't depend on the synthetic fixture's tone/testsrc being
        classified as any one specific type (e.g. "music") -- the point is
        proving demix=True + primary_zone_only=True flow through
        create_content_plan's full 3-tuple return without raising, not
        exercising content-classification accuracy."""
        video = tmp_path / "clip3.mp4"
        _make_test_video(video, duration=4.0)
        monkeypatch.setattr(
            "praisonai_editor.pipeline.OpenAITranscriber",
            lambda: type("T", (), {"transcribe": staticmethod(lambda *a, **k: _fake_transcript())})(),
        )

        result = edit_video(
            str(video), output_path=str(tmp_path / "out3.mp4"),
            preset="no_silence", demix=True, primary_zone_only=True,
            save_artifacts=False,
        )

        assert result.success, result.error
        assert Path(result.output_path).exists()

    def test_save_artifacts_writes_resolved_and_raw_events_like_edit_audio(self, tmp_path, monkeypatch):
        """Parity check: edit_video's content_blocks.json should have the
        same {"resolved": [...], "raw_events": [...]} shape edit_audio's
        does, now that both unpack the same 3-tuple."""
        video = tmp_path / "clip4.mp4"
        _make_test_video(video, duration=4.0)
        monkeypatch.setattr(
            "praisonai_editor.pipeline.OpenAITranscriber",
            lambda: type("T", (), {"transcribe": staticmethod(lambda *a, **k: _fake_transcript())})(),
        )

        result = edit_video(
            str(video), output_path=str(tmp_path / "out4.mp4"),
            preset="no_silence", save_artifacts=True,
        )

        assert result.success, result.error
        blocks_path = Path(result.artifacts["content_blocks"])
        import json
        data = json.loads(blocks_path.read_text())
        assert "resolved" in data and "raw_events" in data


class TestCliWiring:
    """CLI parity gaps found by audit: `edit`'s --min-silence and
    `transcribe`'s --vad-filter existed on the underlying package functions
    but had no CLI flag at all -- only reachable via YAML's generic
    **kwargs passthrough. Mocked at the package-function boundary, same
    convention as test_master.py/test_denoise.py's own CLI wiring tests."""

    def test_edit_min_silence_flag_reaches_edit_media(self, monkeypatch, tmp_path):
        import sys
        import praisonai_editor.cli as cli
        import praisonai_editor.pipeline as pipeline_mod
        from praisonai_editor.models import EditResult

        captured = {}

        def fake_edit_media(input_path, output_path=None, **kwargs):
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path=str(tmp_path / "out.mp3"), success=True)

        monkeypatch.setattr(pipeline_mod, "edit_media", fake_edit_media)
        monkeypatch.setattr(sys, "argv", ["praisonai-editor", "edit", "in.mp3", "--min-silence", "2.5"])
        assert cli.main() == 0
        assert captured["min_silence"] == 2.5

    def test_edit_min_silence_defaults_to_one_point_five(self, monkeypatch, tmp_path):
        import sys
        import praisonai_editor.cli as cli
        import praisonai_editor.pipeline as pipeline_mod
        from praisonai_editor.models import EditResult

        captured = {}

        def fake_edit_media(input_path, output_path=None, **kwargs):
            captured.update(kwargs)
            return EditResult(input_path=input_path, output_path=str(tmp_path / "out.mp3"), success=True)

        monkeypatch.setattr(pipeline_mod, "edit_media", fake_edit_media)
        monkeypatch.setattr(sys, "argv", ["praisonai-editor", "edit", "in.mp3"])
        assert cli.main() == 0
        assert captured["min_silence"] == 1.5

    def test_transcribe_vad_filter_flag_reaches_transcribe_audio(self, monkeypatch, tmp_path):
        import sys
        import praisonai_editor.cli as cli
        import praisonai_editor.transcribe as transcribe_mod
        from praisonai_editor.models import TranscriptResult

        captured = {}

        def fake_transcribe_audio(audio_path, **kwargs):
            captured.update(kwargs)
            return TranscriptResult(text="hi", words=[], language="en", duration=1.0)

        monkeypatch.setattr(transcribe_mod, "transcribe_audio", fake_transcribe_audio)
        monkeypatch.setattr(
            sys, "argv",
            ["praisonai-editor", "transcribe", "in.mp3", "--local", "--vad-filter",
             "-o", str(tmp_path / "out.srt")],
        )
        assert cli.main() == 0
        assert captured["vad_filter"] is True

    def test_transcribe_vad_filter_defaults_to_false(self, monkeypatch, tmp_path):
        import sys
        import praisonai_editor.cli as cli
        import praisonai_editor.transcribe as transcribe_mod
        from praisonai_editor.models import TranscriptResult

        captured = {}

        def fake_transcribe_audio(audio_path, **kwargs):
            captured.update(kwargs)
            return TranscriptResult(text="hi", words=[], language="en", duration=1.0)

        monkeypatch.setattr(transcribe_mod, "transcribe_audio", fake_transcribe_audio)
        monkeypatch.setattr(
            sys, "argv",
            ["praisonai-editor", "transcribe", "in.mp3", "-o", str(tmp_path / "out.srt")],
        )
        assert cli.main() == 0
        assert captured["vad_filter"] is False
