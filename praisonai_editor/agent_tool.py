"""BaseTool subclasses for PraisonAI Agent integration."""

from __future__ import annotations

from typing import Any, Optional


class AudioEditorTool:
    """Edit audio/video files by removing fillers, silences, and repetitions.

    Implements the BaseTool protocol for praisonaiagents auto-discovery.

    Usage:
        from praisonai_editor.agent_tool import audio_editor_tool
        from praisonaiagents import Agent
        agent = Agent(instructions="Audio editor", tools=[audio_editor_tool])
    """

    name = "audio_editor"
    description = "Edit audio/video files by removing filler words, long silences, and repetitions. Supports MP3, MP4, WAV, and other formats."
    version = "1.0.0"

    def run(
        self,
        input_path: str,
        output_path: str = "",
        preset: str = "podcast",
        use_local_whisper: bool = False,
    ) -> dict:
        """Edit a media file.

        Args:
            input_path: Path to the audio/video file to edit
            output_path: Path for the edited output (optional, auto-generated if empty)
            preset: Edit preset - one of: podcast, meeting, course, clean
            use_local_whisper: Use local Whisper model instead of OpenAI API

        Returns:
            Dictionary with editing results including output_path and statistics
        """
        from .pipeline import edit_media

        result = edit_media(
            input_path,
            output_path=output_path or None,
            preset=preset,
            use_local_whisper=use_local_whisper,
            verbose=False,
        )
        return result.to_dict()

    def __call__(self, **kwargs) -> Any:
        return self.run(**kwargs)


class AudioTranscribeTool:
    """Transcribe audio/video files to text with word-level timestamps.

    Implements the BaseTool protocol for praisonaiagents auto-discovery.
    """

    name = "audio_transcribe"
    description = "Transcribe audio or video files to text with word-level timestamps. Returns transcript text and SRT subtitles."
    version = "1.0.0"

    def run(
        self,
        audio_path: str,
        use_local_whisper: bool = False,
        language: str = "",
    ) -> dict:
        """Transcribe a media file.

        Args:
            audio_path: Path to the audio/video file
            use_local_whisper: Use local Whisper model instead of OpenAI API
            language: Language code (e.g. 'en', 'es', 'fr'). Empty for auto-detect.

        Returns:
            Dictionary with text, words (with timestamps), and SRT content
        """
        from .transcribe import transcribe_audio

        result = transcribe_audio(
            audio_path,
            use_local=use_local_whisper,
            language=language or None,
        )
        data = result.to_dict()
        data["srt"] = result.to_srt()
        return data

    def __call__(self, **kwargs) -> Any:
        return self.run(**kwargs)


class MP4ToMP3Tool:
    """Convert MP4 video files to MP3 audio.

    Implements the BaseTool protocol for praisonaiagents auto-discovery.
    """

    name = "mp4_to_mp3"
    description = "Convert MP4 video files to MP3 audio. Also supports WAV, M4A, and other format conversions."
    version = "1.0.0"

    def run(
        self,
        input_path: str,
        output_path: str = "",
        bitrate: str = "192k",
    ) -> dict:
        """Convert a media file.

        Args:
            input_path: Path to the source file (e.g. video.mp4)
            output_path: Path for the converted file (e.g. audio.mp3). If empty, replaces extension with .mp3
            bitrate: Audio bitrate (128k, 192k, 320k)

        Returns:
            Dictionary with the output file path
        """
        from pathlib import Path
        from .convert import convert_media

        if not output_path:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}.mp3")

        result_path = convert_media(input_path, output_path, bitrate=bitrate)
        return {"output_path": result_path, "input_path": input_path}

    def __call__(self, **kwargs) -> Any:
        return self.run(**kwargs)


# Convenience instances
audio_editor_tool = AudioEditorTool()
audio_transcribe_tool = AudioTranscribeTool()
mp4_to_mp3_tool = MP4ToMP3Tool()
