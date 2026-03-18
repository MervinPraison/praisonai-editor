"""PraisonAI Editor — AI-powered audio & video editing.

This package provides:
1. Protocol-driven pipeline: probe → convert → transcribe → plan → render
2. Ready-to-use tools for PraisonAI Agents (AudioEditorTool, etc.)
3. CLI: praisonai-editor edit/transcribe/convert/probe

Usage:
    # Simple audio edit
    from praisonai_editor import edit_audio
    result = edit_audio("podcast.mp3", preset="podcast")

    # Edit any media (auto-detects audio vs video)
    from praisonai_editor import edit_media
    result = edit_media("interview.mp4")

    # Convert MP4 to MP3
    from praisonai_editor import convert_media
    convert_media("video.mp4", "audio.mp3")

    # Transcribe
    from praisonai_editor import transcribe_audio
    transcript = transcribe_audio("podcast.mp3")
    print(transcript.to_srt())

    # Prompt-based AI editing
    from praisonai_editor import prompt_edit
    result = prompt_edit("podcast.mp3", "Remove filler words and the intro")

    # Use with PraisonAI Agents
    from praisonai_editor.agent_tool import audio_editor_tool
    from praisonaiagents import Agent
    agent = Agent(tools=[audio_editor_tool])
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("praisonai-editor")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Mervin Praison"


# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    """Lazy load public symbols."""
    _lazy_map = {
        # Models
        "ProbeResult": ".models",
        "Word": ".models",
        "TranscriptResult": ".models",
        "Segment": ".models",
        "EditPlan": ".models",
        "EditResult": ".models",
        # Protocols
        "Prober": ".protocols",
        "Converter": ".protocols",
        "Transcriber": ".protocols",
        "Editor": ".protocols",
        "Renderer": ".protocols",
        # Probe
        "FFmpegProber": ".probe",
        "probe_media": ".probe",
        # Convert
        "FFmpegConverter": ".convert",
        "convert_media": ".convert",
        # Transcribe
        "OpenAITranscriber": ".transcribe",
        "LocalTranscriber": ".transcribe",
        "transcribe_audio": ".transcribe",
        # Plan
        "HeuristicEditor": ".plan",
        "create_edit_plan": ".plan",
        # Render
        "FFmpegAudioRenderer": ".render",
        "FFmpegVideoRenderer": ".render",
        # Pipeline
        "edit_audio": ".pipeline",
        "edit_video": ".pipeline",
        "edit_media": ".pipeline",
        # Agent pipeline
        "prompt_edit": ".agent_pipeline",
    }

    if name in _lazy_map:
        from importlib import import_module
        module = import_module(_lazy_map[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "ProbeResult",
    "Word",
    "TranscriptResult",
    "Segment",
    "EditPlan",
    "EditResult",
    # Protocols
    "Prober",
    "Converter",
    "Transcriber",
    "Editor",
    "Renderer",
    # Core functions
    "probe_media",
    "convert_media",
    "transcribe_audio",
    "create_edit_plan",
    # Pipeline
    "edit_audio",
    "edit_video",
    "edit_media",
    "prompt_edit",
    # Implementations
    "FFmpegProber",
    "FFmpegConverter",
    "OpenAITranscriber",
    "LocalTranscriber",
    "HeuristicEditor",
    "FFmpegAudioRenderer",
    "FFmpegVideoRenderer",
]
