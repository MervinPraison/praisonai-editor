"""Self-describing source-audio tags on a transcript.

A transcript produced by :func:`praisonai_editor.transcribe.transcribe_audio`
carries a ``source_audio`` tag (filename, size, content fingerprint,
duration) identifying the exact audio file it was made from. This lets a
transcript that arrives on its own -- separated from its audio, e.g. moved
to a different folder or handed to a tool that only accepts one file --
be matched back to its audio automatically, instead of relying on both
files sharing a directory/name convention.

The fingerprint reuses the same convention as the Demucs vocal-isolation
cache (:mod:`praisonai_editor._demix`): SHA-256 of the first 8 MiB, hex,
truncated to 16 chars. Hashing only the first 8 MiB keeps tagging and
matching cheap even for hour-long files, while still being exact for
byte-identical files (a real audio codec header differs within the first
few KB for almost any distinct recording).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Same convention as _demix.py's cache key -- see module docstring.
_HASH_READ_BYTES = 8 * 1024 * 1024

#: Extensions considered when scanning a directory for candidate audio.
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

#: Default tolerance (seconds) for a duration-based match when content
#: hashes don't line up (e.g. the audio was re-encoded after transcription).
DEFAULT_DURATION_TOLERANCE = 0.5


def _content_fingerprint(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(_HASH_READ_BYTES)).hexdigest()[:16]


def tag_source_audio(result: Any, audio_path: str) -> Any:
    """Stamp ``result.source_audio`` in place and return *result*.

    Shared by every transcription entry point (the ``transcribe_audio()``
    convenience function, and the pipeline/agent callers that use
    ``OpenAITranscriber``/``LocalTranscriber`` directly) so the tag is
    never missing just because a caller took a different path in. Silently
    skips tagging (leaving ``source_audio`` unset) if *audio_path* is no
    longer readable -- the tag is a convenience for later re-matching, not
    something transcription itself should fail over.
    """
    try:
        result.source_audio = compute_audio_tag(audio_path)
    except OSError:
        pass
    return result


def compute_audio_tag(audio_path: str) -> Dict[str, Any]:
    """Build the ``source_audio`` tag to stamp onto a transcript.

    Raises FileNotFoundError/OSError the same way any of the callers'
    existing file access does -- not caught here, since a transcript
    produced from an unreadable audio file is already a hard failure
    upstream of this call.
    """
    real = os.path.realpath(audio_path)
    return {
        "filename": os.path.basename(real),
        "size_bytes": os.path.getsize(real),
        "sha256_8mb": _content_fingerprint(real),
    }


def find_matching_audio(
    source_audio: Dict[str, Any],
    search_dirs: List[str],
    *,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = DEFAULT_DURATION_TOLERANCE,
) -> List[Dict[str, Any]]:
    """Search *search_dirs* (top-level, non-recursive) for the audio a
    transcript's ``source_audio`` tag describes.

    Returns candidates ranked strongest-first:
      - ``"exact"``: content fingerprint AND size both match -- effectively
        certain to be the same file (or a byte-identical copy).
      - ``"filename"``: same filename, found in a different directory than
        where the transcript looked (fingerprint didn't match -- the file
        was likely modified/re-encoded since transcription).
      - ``"duration"``: *expected_duration* given and a candidate's probed
        duration is within *duration_tolerance* seconds, filename differs.
        Only attempted when *expected_duration* is provided, since probing
        every candidate's duration is comparatively expensive.

    Each candidate dict: ``{"path": str, "confidence": str, "filename": str}``.
    Empty list if nothing matches. Unreadable directories are skipped, not
    raised -- a caller scanning several known library dirs shouldn't fail
    outright because one of them doesn't exist on this machine.
    """
    if not source_audio:
        return []

    target_name = source_audio.get("filename")
    target_size = source_audio.get("size_bytes")
    target_hash = source_audio.get("sha256_8mb")

    exact: List[Dict[str, Any]] = []
    by_filename: List[Dict[str, Any]] = []
    by_duration: List[Dict[str, Any]] = []
    seen_paths = set()

    for d in search_dirs:
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for fname in entries:
            if not fname.lower().endswith(AUDIO_EXTS):
                continue
            candidate = os.path.join(d, fname)
            real = os.path.realpath(candidate)
            if not os.path.isfile(real) or real in seen_paths:
                continue

            if target_hash and target_size is not None:
                try:
                    if (
                        os.path.getsize(real) == target_size
                        and _content_fingerprint(real) == target_hash
                    ):
                        exact.append({"path": real, "confidence": "exact", "filename": fname})
                        seen_paths.add(real)
                        continue
                except OSError:
                    pass

            if target_name and fname == target_name:
                by_filename.append({"path": real, "confidence": "filename", "filename": fname})
                seen_paths.add(real)
                continue

            if expected_duration is not None:
                duration = _probe_duration_safe(real)
                if duration is not None and abs(duration - expected_duration) <= duration_tolerance:
                    by_duration.append({"path": real, "confidence": "duration", "filename": fname})
                    seen_paths.add(real)

    return exact + by_filename + by_duration


def _probe_duration_safe(path: str) -> Optional[float]:
    try:
        from .probe import probe_media
    except ImportError:
        return None
    try:
        return probe_media(path).duration
    except Exception:
        return None
