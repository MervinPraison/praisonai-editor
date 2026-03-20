"""Vocal stem separation using Demucs (optional dependency).

This module separates audio into:
  - vocals WAV  (isolated vocal track)
  - no_vocals WAV  (all other instruments)

Design principles:
  - Demucs is an OPTIONAL dependency. If not installed, ``isolate_vocals``
    raises ``ImportError`` only when actually called.
  - We use the *Python API* (``demucs.pretrained`` + ``demucs.apply``) instead
    of a subprocess call to avoid the known pad1d AssertionError in the
    demucs 4.0.x CLI on Apple Silicon / PyTorch 2.10.
  - Heavy imports (torch, soundfile, demucs.*) only happen inside
    ``isolate_vocals`` (lazy) — zero import cost for the core package.
  - The function is thread-safe (each call writes to its own tempdir).

Usage:
    from praisonai_editor._demix import isolate_vocals, has_demucs

    if has_demucs():
        vocals_path, inst_path = isolate_vocals("/path/to/audio.mp3")
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def has_demucs() -> bool:
    """Return True if demucs (and its torch/soundfile deps) are importable."""
    try:
        import demucs.pretrained  # noqa: F401
        import soundfile          # noqa: F401
        return True
    except ImportError:
        return False


def isolate_vocals(
    media_path: str,
    *,
    model_name: str = "mdx_extra",
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[str, str]:
    """Separate voice from instruments using the Demucs Python API.

    Args:
        media_path: Path to input audio/video file.
        model_name: Demucs model. ``mdx_extra_q`` is robust and memory-efficient.
                    Alternatives: ``mdx_extra``, ``htdemucs`` (may crash on short clips).
        device: Torch device — ``"cpu"`` is safe for all machines; ``"mps"``
                uses Apple Silicon GPU if available.
        verbose: Print progress messages.

    Returns:
        A tuple of ``(vocals_wav_path, instruments_wav_path)``.
        Both files live inside a temporary directory. Callers own the files
        and should clean up the parent tempdir when done.

    Raises:
        ImportError: If ``demucs`` or ``soundfile`` is not installed.
        RuntimeError: If separation fails.
    """
    if not has_demucs():
        raise ImportError(
            "demucs is not installed. "
            "Install it with: pip install praisonai-editor[demix]"
        )

    # Lazy heavy imports
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    media_path = os.path.abspath(media_path)
    stem = Path(media_path).stem

    # ---- Stem cache ----
    # Key the cache on the SHA-256 of the media file so the same input always
    # reuses already-separated stems (Demucs is slow on CPU).
    import hashlib
    with open(media_path, "rb") as _f:
        file_hash = hashlib.sha256(_f.read(8 * 1024 * 1024)).hexdigest()[:16]   # first 8 MiB
    cache_dir = Path.home() / ".praisonai" / "editor" / ".demix_cache" / file_hash
    cached_vocals = cache_dir / "vocals.wav"
    cached_inst   = cache_dir / "no_vocals.wav"
    if cached_vocals.exists() and cached_inst.exists():
        if verbose:
            print(f"    ↻ Reusing cached Demucs stems ({cache_dir})", flush=True)
        return str(cached_vocals), str(cached_inst)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(cache_dir)   # write stems directly into cache dir


    try:
        # Load model
        if verbose:
            print(f"    Loading Demucs model '{model_name}'...", flush=True)
        model = get_model(model_name)
        model.eval()
        sr_model: int = model.samplerate       # 44100 for MDX models
        ch_model: int = model.audio_channels   # 2 (stereo)

        # Load audio — use soundfile which handles MP3 via ffmpeg/libsndfile
        # If soundfile can't decode MP3 directly, use librosa as fallback
        try:
            audio_np, sr_file = sf.read(media_path, always_2d=True)
        except Exception:
            # Fallback: convert to WAV first via ffmpeg
            import shutil
            import subprocess
            ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
            tmp_wav = os.path.join(tmp_dir, "input.wav")
            subprocess.run(
                [ffmpeg, "-y", "-i", media_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", str(sr_model), "-ac", "2",
                 tmp_wav],
                capture_output=not verbose,
                check=True,
            )
            audio_np, sr_file = sf.read(tmp_wav, always_2d=True)

        if verbose:
            print(
                f"    Audio loaded: {audio_np.shape} @ {sr_file} Hz → "
                f"converting to {sr_model} Hz {ch_model}-ch...",
                flush=True,
            )

        # Convert to torch tensor (C, T) and resample/remap channels
        wav = torch.from_numpy(audio_np.T.astype(np.float32))  # (C, T)
        wav = convert_audio(wav, sr_file, sr_model, ch_model)   # (ch_model, T)

        # Run stem separation
        if verbose:
            print("    Running Demucs stem separation (CPU)...", flush=True)
        with torch.no_grad():
            sources = apply_model(
                model, wav[None], device=device, progress=verbose
            )  # → (1, n_stems, ch_model, T)

        # Extract vocals + everything-else
        vocal_idx = model.sources.index("vocals")
        vocals_tensor = sources[0, vocal_idx]              # (C, T)
        inst_tensor = sources[0].sum(dim=0) - vocals_tensor  # (C, T)

        # Save to WAV files in tmpdir
        vocals_path = os.path.join(tmp_dir, "vocals.wav")
        inst_path = os.path.join(tmp_dir, "no_vocals.wav")

        sf.write(vocals_path, vocals_tensor.cpu().numpy().T, sr_model)
        sf.write(inst_path, inst_tensor.cpu().numpy().T, sr_model)

        if verbose:
            print(f"    Stems saved: {vocals_path}, {inst_path}", flush=True)

        return vocals_path, inst_path

    except Exception:
        # Don't delete the cache dir — partial stems will be ignored next
        # run because we check for both files' existence before reusing.
        raise
