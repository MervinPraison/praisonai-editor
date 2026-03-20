"""Audio transcription with word-level timestamps."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .models import TranscriptResult, Word

# Max file size for Whisper API (24MB with safety margin)
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
# Chunk duration in seconds (~10 minutes → fits within 24MB as MP3)
CHUNK_DURATION_SECS = 600
# Force chunking when audio exceeds this duration (regardless of file size).
# 40-min audio compresses to ~19 MB (under 25 MB limit) but times out API.
MAX_AUDIO_DURATION_SECS = 600


def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("ffmpeg not found")


def _extract_audio_mp3(media_path: str, output_path: str) -> None:
    """Extract audio from media file to compressed MP3 (mono 64k) for upload."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", media_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ab", "64k",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _extract_audio_wav(media_path: str, output_path: str) -> None:
    """Extract audio from media file to WAV (16kHz mono) for Whisper."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", media_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _split_audio(audio_path: str, chunk_dir: str, chunk_secs: int = CHUNK_DURATION_SECS) -> List[str]:
    """Split audio into chunks of chunk_secs duration."""
    ffmpeg = _find_ffmpeg()
    # Get duration
    probe_cmd = [
        shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        audio_path,
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    duration = float(json.loads(result.stdout)["format"]["duration"])

    n_chunks = math.ceil(duration / chunk_secs)
    chunk_paths = []

    for i in range(n_chunks):
        start = i * chunk_secs
        chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.mp3")
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start),
            "-i", audio_path,
            "-t", str(chunk_secs),
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunk_paths.append(chunk_path)

    return chunk_paths


class OpenAITranscriber:
    """Transcribes audio using OpenAI Whisper API. Implements the Transcriber protocol.

    Automatically handles large files by:
    1. Extracting audio to compressed MP3 (mono 64kbps)
    2. If still >24MB, splitting into ~10-minute chunks
    3. Transcribing each chunk and merging results with corrected timestamps
    """

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        model: str = "whisper-1",
    ) -> TranscriptResult:
        """Transcribe using the OpenAI Whisper API.

        Requires OPENAI_API_KEY environment variable.
        Handles files of any size via automatic chunking.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            model: Whisper model name

        Returns:
            TranscriptResult with word-level timestamps
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable required")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Extract audio to compressed MP3 (always — keeps size minimal)
            mp3_path = os.path.join(tmpdir, "audio.mp3")
            _extract_audio_mp3(str(path), mp3_path)

            file_size = os.path.getsize(mp3_path)

            # Estimate duration from file size (mono 64 kbps = 8000 bytes/s)
            estimated_secs = file_size / 8000

            # Step 2: If small enough AND short enough, transcribe directly.
            # Long files (>10 min) time out the API even if they fit in 25 MB.
            if file_size <= MAX_UPLOAD_BYTES and estimated_secs <= MAX_AUDIO_DURATION_SECS:
                return self._call_api(mp3_path, model, language)

            # Step 3: Split into chunks and transcribe each
            chunk_dir = os.path.join(tmpdir, "chunks")
            os.makedirs(chunk_dir)
            chunk_paths = _split_audio(mp3_path, chunk_dir)

            all_words: List[Word] = []
            all_texts: List[str] = []
            total_duration = 0.0
            detected_language = "en"

            for i, chunk_path in enumerate(chunk_paths):
                chunk_offset = i * CHUNK_DURATION_SECS
                # Skip empty or near-empty chunk files (last chunk may be <0.1s)
                # 1289-byte chunk = MP3 header only (no real audio) → "audio_too_short" error
                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) < 5000:
                    continue
                result = self._call_api(chunk_path, model, language)

                all_texts.append(result.text)
                detected_language = result.language

                # Offset word timestamps by chunk start time
                for w in result.words:
                    all_words.append(Word(
                        text=w.text,
                        start=w.start + chunk_offset,
                        end=w.end + chunk_offset,
                        confidence=w.confidence,
                    ))

                total_duration = max(total_duration, chunk_offset + result.duration)

            return TranscriptResult(
                text=" ".join(all_texts),
                words=all_words,
                language=detected_language,
                duration=total_duration,
            )

    def _call_api(
        self,
        audio_path: str,
        model: str,
        language: str | None,
    ) -> TranscriptResult:
        from openai import OpenAI

        # Generous timeout: 10-min chunk can take a while to upload+process
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=600.0)

        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language=language,
            )

        words = []
        if hasattr(response, "words") and response.words:
            for w in response.words:
                words.append(
                    Word(
                        text=w.word,
                        start=w.start,
                        end=w.end,
                        confidence=1.0,
                    )
                )

        return TranscriptResult(
            text=response.text,
            words=words,
            language=response.language if hasattr(response, "language") else "en",
            duration=response.duration if hasattr(response, "duration") else 0.0,
        )


class LocalTranscriber:
    """Transcribes audio using local faster-whisper. Implements the Transcriber protocol."""

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        model: str = "base",
    ) -> TranscriptResult:
        """Transcribe using local faster-whisper.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            model: Model name (tiny, base, small, medium, large-v3)

        Returns:
            TranscriptResult with word-level timestamps
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("faster-whisper required: pip install 'praisonai-editor[local]'")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Extract to WAV if needed
        if path.suffix.lower() not in (".wav",):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                _extract_audio_wav(str(path), wav_path)
                return self._run_whisper(wav_path, model, language)
            finally:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
        else:
            return self._run_whisper(str(path), model, language)

    def _run_whisper(
        self,
        audio_path: str,
        model: str,
        language: str | None,
    ) -> TranscriptResult:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel(model, device="auto", compute_type="auto")
        segments, info = whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
        )

        words = []
        full_text = []

        for segment in segments:
            full_text.append(segment.text)
            if segment.words:
                for w in segment.words:
                    words.append(
                        Word(
                            text=w.word,
                            start=w.start,
                            end=w.end,
                            confidence=w.probability,
                        )
                    )

        return TranscriptResult(
            text=" ".join(full_text),
            words=words,
            language=info.language,
            duration=info.duration,
        )


# Module-level convenience function
def transcribe_audio(
    audio_path: str,
    *,
    use_local: bool = False,
    language: str | None = None,
    model: str | None = None,
) -> TranscriptResult:
    """Transcribe an audio or video file.

    Args:
        audio_path: Path to audio/video file
        use_local: If True, use local faster-whisper
        language: Optional language code
        model: Model name (default: whisper-1 for API, base for local)

    Returns:
        TranscriptResult with word-level timestamps
    """
    if use_local:
        transcriber = LocalTranscriber()
        return transcriber.transcribe(
            audio_path,
            language=language,
            model=model or "base",
        )
    else:
        transcriber = OpenAITranscriber()
        return transcriber.transcribe(
            audio_path,
            language=language,
            model=model or "whisper-1",
        )
