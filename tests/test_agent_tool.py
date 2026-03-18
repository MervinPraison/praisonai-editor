"""Tests for agent tool classes."""

from praisonai_editor.agent_tool import (
    AudioEditorTool, AudioTranscribeTool, MP4ToMP3Tool,
    audio_editor_tool, audio_transcribe_tool, mp4_to_mp3_tool,
)


class TestAudioEditorTool:
    def test_has_name(self):
        tool = AudioEditorTool()
        assert tool.name == "audio_editor"
        assert len(tool.description) > 10

    def test_is_callable(self):
        tool = AudioEditorTool()
        assert callable(tool)


class TestAudioTranscribeTool:
    def test_has_name(self):
        tool = AudioTranscribeTool()
        assert tool.name == "audio_transcribe"


class TestMP4ToMP3Tool:
    def test_has_name(self):
        tool = MP4ToMP3Tool()
        assert tool.name == "mp4_to_mp3"


class TestConvenienceInstances:
    def test_instances_exist(self):
        assert audio_editor_tool is not None
        assert audio_transcribe_tool is not None
        assert mp4_to_mp3_tool is not None

    def test_instances_have_names(self):
        assert audio_editor_tool.name == "audio_editor"
        assert audio_transcribe_tool.name == "audio_transcribe"
        assert mp4_to_mp3_tool.name == "mp4_to_mp3"
