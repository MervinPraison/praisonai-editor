"""Shared data classes for the PraisonAI Editor pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProbeResult:
    """Result of probing a media file."""

    path: str
    duration: float
    has_video: bool = False
    # Audio info
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    audio_bitrate: Optional[int] = None
    # Video info (only if has_video)
    video_codec: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    # General
    format_name: str = ""
    size_bytes: int = 0
    bit_rate: int = 0
    streams: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "duration": self.duration,
            "has_video": self.has_video,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "audio_bitrate": self.audio_bitrate,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format_name": self.format_name,
            "size_bytes": self.size_bytes,
            "bit_rate": self.bit_rate,
        }

    @property
    def is_audio_only(self) -> bool:
        return not self.has_video


@dataclass
class Word:
    """A word with timing information."""

    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptResult:
    """Result of transcription."""

    text: str
    words: List[Word] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptResult":
        words = [
            Word(
                text=str(w.get("text", "")),
                start=float(w["start"]),
                end=float(w["end"]),
                confidence=float(w.get("confidence", 1.0)),
            )
            for w in data.get("words", [])
        ]
        return cls(
            text=str(data.get("text", "") or ""),
            words=words,
            language=str(data.get("language", "en") or "en"),
            duration=float(data.get("duration", 0.0)),
        )

    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        if not self.words:
            return ""

        lines: List[str] = []
        idx = 1
        segment_words: List[Word] = []

        for word in self.words:
            segment_words.append(word)
            if word.text.rstrip().endswith((".", "!", "?", ",")) or len(segment_words) >= 7:
                if segment_words:
                    start = segment_words[0].start
                    end = segment_words[-1].end
                    text = " ".join(w.text for w in segment_words)
                    lines.append(f"{idx}")
                    lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
                    lines.append(text.strip())
                    lines.append("")
                    idx += 1
                    segment_words = []

        if segment_words:
            start = segment_words[0].start
            end = segment_words[-1].end
            text = " ".join(w.text for w in segment_words)
            lines.append(f"{idx}")
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(text.strip())
            lines.append("")

        return "\n".join(lines)


@dataclass
class Segment:
    """A segment of media to keep or remove."""

    start: float
    end: float
    action: str  # "keep" or "remove"
    reason: str
    category: str  # "filler", "repetition", "silence", "tangent", "content"
    text: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "action": self.action,
            "reason": self.reason,
            "category": self.category,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class EditPlan:
    """Complete edit plan for a media file."""

    segments: List[Segment] = field(default_factory=list)
    original_duration: float = 0.0
    edited_duration: float = 0.0
    removed_duration: float = 0.0
    removal_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "original_duration": self.original_duration,
            "edited_duration": self.edited_duration,
            "removed_duration": self.removed_duration,
            "removal_summary": self.removal_summary,
        }

    def get_keep_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.action == "keep"]

    def get_remove_segments(self) -> List[Segment]:
        return [s for s in self.segments if s.action == "remove"]


@dataclass
class EditResult:
    """Result of a media editing operation."""

    input_path: str
    output_path: str
    probe: Optional[ProbeResult] = None
    transcript: Optional[TranscriptResult] = None
    plan: Optional[EditPlan] = None
    success: bool = True
    error: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "probe": self.probe.to_dict() if self.probe else None,
            "transcript": self.transcript.to_dict() if self.transcript else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "success": self.success,
            "error": self.error,
            "artifacts": self.artifacts,
        }


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
