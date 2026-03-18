"""Protocol interfaces for the PraisonAI Editor pipeline.

Each pipeline stage is defined as a Protocol class, enabling custom implementations
to be swapped in without modifying core logic.

Usage:
    from praisonai_editor.protocols import Transcriber

    class MyTranscriber:
        def transcribe(self, audio_path: str, **kwargs) -> TranscriptResult:
            ...  # custom implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import EditPlan, ProbeResult, TranscriptResult


@runtime_checkable
class Prober(Protocol):
    """Protocol for probing media files to extract metadata."""

    def probe(self, path: str) -> ProbeResult:
        """Probe a media file and return metadata.

        Args:
            path: Path to the media file (MP3, MP4, WAV, etc.)

        Returns:
            ProbeResult with duration, codec, channels, etc.
        """
        ...


@runtime_checkable
class Converter(Protocol):
    """Protocol for converting between media formats."""

    def convert(
        self,
        input_path: str,
        output_path: str,
        *,
        bitrate: str = "192k",
    ) -> str:
        """Convert a media file to another format.

        Args:
            input_path: Path to source file
            output_path: Path for converted file
            bitrate: Audio bitrate (e.g. "128k", "192k", "320k")

        Returns:
            Path to the converted file
        """
        ...


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for transcribing audio to text with word-level timestamps."""

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
    ) -> TranscriptResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g. "en")

        Returns:
            TranscriptResult with text and word-level timestamps
        """
        ...


@runtime_checkable
class Editor(Protocol):
    """Protocol for creating edit plans from transcripts."""

    def create_plan(
        self,
        transcript: TranscriptResult,
        duration: float,
        *,
        remove_fillers: bool = True,
        remove_repetitions: bool = True,
        remove_silence: bool = True,
        min_silence: float = 1.5,
    ) -> EditPlan:
        """Create an edit plan based on transcript analysis.

        Args:
            transcript: Transcription result with word-level timestamps
            duration: Total media duration in seconds
            remove_fillers: Remove filler words (um, uh, etc.)
            remove_repetitions: Remove repeated words
            remove_silence: Remove long silences
            min_silence: Minimum silence duration to remove

        Returns:
            EditPlan with segments to keep/remove
        """
        ...


@runtime_checkable
class Renderer(Protocol):
    """Protocol for rendering edited media from an edit plan."""

    def render(
        self,
        input_path: str,
        output_path: str,
        plan: EditPlan,
        *,
        copy_codec: bool = True,
    ) -> str:
        """Render media based on an edit plan.

        Args:
            input_path: Path to original media file
            output_path: Path for rendered output
            plan: EditPlan with segments to keep
            copy_codec: If True, copy codecs (faster). If False, re-encode.

        Returns:
            Path to rendered file
        """
        ...
