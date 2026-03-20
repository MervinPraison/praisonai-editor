"""Full media editing pipeline — orchestrates probe → convert → transcribe → plan → render."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import EditPlan, EditResult, ProbeResult, TranscriptResult, Word
from .probe import FFmpegProber
from .convert import FFmpegConverter
from .transcribe import OpenAITranscriber, LocalTranscriber
from .plan import HeuristicEditor, get_preset_config, PRESETS
from .render import FFmpegAudioRenderer, FFmpegVideoRenderer

# Content-based presets that use detect.py
CONTENT_PRESETS = {"songs_only", "speech_only", "no_silence"}

# Audio-only extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".opus"}
# Video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


def _load_cached_transcript(artifacts_dir: Path, verbose: bool = False) -> Optional[TranscriptResult]:
    """Load transcript from cached artifacts if available."""
    json_path = artifacts_dir / "transcript.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)

        words = [
            Word(text=w["text"], start=w["start"], end=w["end"], confidence=w.get("confidence", 1.0))
            for w in data.get("words", [])
        ]
        transcript = TranscriptResult(
            text=data.get("text", ""),
            words=words,
            language=data.get("language", "en"),
            duration=data.get("duration", 0.0),
        )
        if verbose:
            print(f"    ↻ Reusing cached transcript ({len(words)} words)", flush=True)
        return transcript
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def edit_media(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    preset: str = "podcast",
    detector: str = "auto",
    demix: bool = False,
    primary_zone_only: bool = False,
    remove_fillers: bool = True,
    remove_repetitions: bool = True,
    remove_silence: bool = True,
    min_silence: float = 1.5,
    use_local_whisper: bool = False,
    language: Optional[str] = None,
    copy_codec: bool = True,
    verbose: bool = False,
    save_artifacts: bool = True,
) -> EditResult:
    """Edit any media file — auto-detects audio vs video and routes accordingly.

    Args:
        input_path: Path to input media file (MP3, MP4, etc.)
        output_path: Path for output (default: {input}_edited.{ext})
        preset: Edit preset (podcast, meeting, course, clean, songs_only, speech_only, no_silence)
        remove_fillers: Remove filler words
        remove_repetitions: Remove repeated words
        remove_silence: Remove long silences
        min_silence: Minimum silence duration to remove
        use_local_whisper: Use local faster-whisper instead of OpenAI API
        language: Language code for transcription (e.g., 'ta' for Tamil)
        copy_codec: Copy codecs (faster) vs re-encode
        verbose: Print progress
        save_artifacts: Save transcript, plan, etc. as files

    Returns:
        EditResult with all outputs and artifacts
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_file.suffix.lower()

    if ext in VIDEO_EXTENSIONS:
        return edit_video(
            input_path, output_path,
            preset=preset,
            detector=detector,
            demix=demix,
            primary_zone_only=primary_zone_only,
            remove_fillers=remove_fillers,
            remove_repetitions=remove_repetitions,
            remove_silence=remove_silence,
            min_silence=min_silence,
            use_local_whisper=use_local_whisper,
            language=language,
            copy_codec=copy_codec,
            verbose=verbose,
            save_artifacts=save_artifacts,
        )
    else:
        return edit_audio(
            input_path, output_path,
            preset=preset,
            detector=detector,
            demix=demix,
            primary_zone_only=primary_zone_only,
            remove_fillers=remove_fillers,
            remove_repetitions=remove_repetitions,
            remove_silence=remove_silence,
            min_silence=min_silence,
            use_local_whisper=use_local_whisper,
            language=language,
            copy_codec=copy_codec,
            verbose=verbose,
            save_artifacts=save_artifacts,
        )


def edit_audio(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    preset: str = "podcast",
    detector: str = "auto",
    demix: bool = False,
    primary_zone_only: bool = False,
    remove_fillers: bool = True,
    remove_repetitions: bool = True,
    remove_silence: bool = True,
    min_silence: float = 1.5,
    use_local_whisper: bool = False,
    language: Optional[str] = None,
    copy_codec: bool = True,
    verbose: bool = False,
    save_artifacts: bool = True,
) -> EditResult:
    """Edit an audio file: transcribe → plan → render.

    Args:
        input_path: Path to audio file (MP3, WAV, etc.)
        output_path: Path for output audio
        preset: Edit preset name (podcast/meeting/course/clean/songs_only/speech_only/no_silence)
        language: Language code for transcription (e.g. 'ta' for Tamil)
        Other args: see edit_media()

    Returns:
        EditResult
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = str(input_file.parent / f"{input_file.stem}_edited{input_file.suffix}")

    output_file = Path(output_path)
    artifacts_dir = Path.home() / f".praisonai/editor/{input_file.stem}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}

    # Determine if this is a content-based preset
    use_content_detection = preset in CONTENT_PRESETS

    # Apply heuristic preset config (only for non-content presets)
    if not use_content_detection:
        config = get_preset_config(preset)
        if config:
            remove_fillers = config.get("remove_fillers", remove_fillers)
            remove_repetitions = config.get("remove_repetitions", remove_repetitions)
            remove_silence = config.get("remove_silence", remove_silence)
            min_silence = config.get("min_silence", min_silence)

    probe = None
    transcript = None
    plan = None

    try:
        # Step 1: Probe
        if verbose:
            print(f"[1/3] Probing: {input_path}", flush=True)
        prober = FFmpegProber()
        probe = prober.probe(input_path)

        if save_artifacts:
            probe_path = artifacts_dir / "probe.json"
            with open(probe_path, "w") as f:
                json.dump(probe.to_dict(), f, indent=2)
            artifacts["probe"] = str(probe_path)

        # Step 2: Transcribe (use cache if available)
        transcript = _load_cached_transcript(artifacts_dir, verbose)
        if transcript is None:
            if verbose:
                print(f"[2/3] Transcribing ({probe.duration:.1f}s)...", flush=True)
            if use_local_whisper:
                transcriber = LocalTranscriber()
            else:
                transcriber = OpenAITranscriber()
            transcript = transcriber.transcribe(input_path, language=language)
        else:
            if verbose:
                print("[2/3] Transcript loaded from cache", flush=True)

        if save_artifacts:
            txt_path = artifacts_dir / "transcript.txt"
            with open(txt_path, "w") as f:
                f.write(transcript.text)
            artifacts["transcript_txt"] = str(txt_path)

            srt_path = artifacts_dir / "transcript.srt"
            with open(srt_path, "w") as f:
                f.write(transcript.to_srt())
            artifacts["transcript_srt"] = str(srt_path)

            json_path = artifacts_dir / "transcript.json"
            with open(json_path, "w") as f:
                json.dump(transcript.to_dict(), f, indent=2)
            artifacts["transcript_json"] = str(json_path)

        if verbose:
            print("[3/3] Creating edit plan and rendering...", flush=True)

        if use_content_detection:
            from .detect import create_content_plan
            # When demix is enabled, speech_over_music has been precisely split into
            # 'singing' vs 'talking_over_music' — so songs_only should ONLY keep 'singing' and 'music'.
            # Without demix, keep speech_over_music too (it hasn't been split).
            keep_map = {
                "songs_only": ["music", "singing"] if demix else ["music", "singing", "speech_over_music"],
                "speech_only": ["speech", "talking_over_music", "speech_over_music"],
                "no_silence": ["speech", "music", "singing", "talking_over_music", "speech_over_music"],
            }
            plan, blocks, all_events = create_content_plan(
                input_path, transcript, probe.duration,
                keep_types=keep_map[preset],
                detector=detector,
                demix=demix,
                primary_zone_only=primary_zone_only,
                verbose=verbose,
            )
            # Save content detection results
            if save_artifacts:
                def _b_dict(b):
                    return {
                        "start": b.start, "end": b.end, "duration": b.duration,
                        "type": b.content_type, "detector": b.detector,
                        "rms_db": round(b.mean_volume, 1),
                        "crest_factor": round(b.crest_factor, 1),
                        "dynamic_range": round(b.dynamic_range, 1),
                        "zero_crossing_rate": round(b.zero_crossing_rate, 4),
                        "confidence": round(b.confidence, 2)
                    }
                blocks_data = {
                    "resolved": [_b_dict(b) for b in blocks],
                    "raw_events": [_b_dict(e) for e in all_events]
                }
                blocks_path = artifacts_dir / "content_blocks.json"
                artifacts_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists
                with open(blocks_path, "w") as f:
                    json.dump(blocks_data, f, indent=2)
                artifacts["content_blocks"] = str(blocks_path)
        else:
            # Heuristic editing (podcast, meeting, course, clean)
            editor = HeuristicEditor()
            plan = editor.create_plan(
                transcript, probe.duration,
                remove_fillers=remove_fillers,
                remove_repetitions=remove_repetitions,
                remove_silence=remove_silence,
                min_silence=min_silence,
            )

        if save_artifacts:
            plan_path = artifacts_dir / "plan.json"
            with open(plan_path, "w") as f:
                json.dump(plan.to_dict(), f, indent=2)
            artifacts["plan"] = str(plan_path)

        if verbose:
            print(f"    Original: {plan.original_duration:.1f}s", flush=True)
            print(f"    Edited:   {plan.edited_duration:.1f}s", flush=True)
            print(f"    Removed:  {plan.removed_duration:.1f}s ({plan.removed_duration / plan.original_duration * 100:.1f}%)", flush=True)
            for cat, dur in plan.removal_summary.items():
                print(f"      - {cat}: {dur:.1f}s", flush=True)

        renderer = FFmpegAudioRenderer()
        renderer.render(input_path, output_path, plan, copy_codec=copy_codec, verbose=verbose)
        artifacts["output"] = output_path

        if verbose:
            print(f"\n✓ Done! Output: {output_path}", flush=True)

        return EditResult(
            input_path=input_path,
            output_path=output_path,
            probe=probe,
            transcript=transcript,
            plan=plan,
            success=True,
            artifacts=artifacts,
        )

    except Exception as e:
        import traceback as _tb
        _tb.print_exc()   # log the full stack to stderr for debugging
        return EditResult(
            input_path=input_path,
            output_path=output_path,
            probe=probe,
            transcript=transcript,
            plan=plan,
            success=False,
            error=str(e),
            artifacts=artifacts,
        )


def edit_video(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    preset: str = "podcast",
    detector: str = "auto",
    remove_fillers: bool = True,
    remove_repetitions: bool = True,
    remove_silence: bool = True,
    min_silence: float = 1.5,
    use_local_whisper: bool = False,
    language: Optional[str] = None,
    copy_codec: bool = True,
    verbose: bool = False,
    save_artifacts: bool = True,
) -> EditResult:
    """Edit a video file: transcribe audio → plan → render video."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = str(input_file.parent / f"{input_file.stem}_edited{input_file.suffix}")

    output_file = Path(output_path)
    artifacts_dir = Path.home() / f".praisonai/editor/{input_file.stem}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}

    use_content_detection = preset in CONTENT_PRESETS

    if not use_content_detection:
        config = get_preset_config(preset)
        if config:
            remove_fillers = config.get("remove_fillers", remove_fillers)
            remove_repetitions = config.get("remove_repetitions", remove_repetitions)
            remove_silence = config.get("remove_silence", remove_silence)
            min_silence = config.get("min_silence", min_silence)

    probe = None
    transcript = None
    plan = None

    try:
        # Step 1: Probe
        if verbose:
            print(f"[1/4] Probing: {input_path}", flush=True)
        prober = FFmpegProber()
        probe = prober.probe(input_path)

        if save_artifacts:
            probe_path = artifacts_dir / "probe.json"
            with open(probe_path, "w") as f:
                json.dump(probe.to_dict(), f, indent=2)
            artifacts["probe"] = str(probe_path)

        # Step 2: Transcribe (use cache if available)
        transcript = _load_cached_transcript(artifacts_dir, verbose)
        if transcript is None:
            if verbose:
                print(f"[2/4] Transcribing ({probe.duration:.1f}s)...", flush=True)
            if use_local_whisper:
                transcriber = LocalTranscriber()
            else:
                transcriber = OpenAITranscriber()
            transcript = transcriber.transcribe(input_path, language=language)
        else:
            if verbose:
                print("[2/4] Transcript loaded from cache", flush=True)

        if save_artifacts:
            txt_path = artifacts_dir / "transcript.txt"
            with open(txt_path, "w") as f:
                f.write(transcript.text)
            artifacts["transcript_txt"] = str(txt_path)

            srt_path = artifacts_dir / "transcript.srt"
            with open(srt_path, "w") as f:
                f.write(transcript.to_srt())
            artifacts["transcript_srt"] = str(srt_path)

            json_path = artifacts_dir / "transcript.json"
            with open(json_path, "w") as f:
                json.dump(transcript.to_dict(), f, indent=2)
            artifacts["transcript_json"] = str(json_path)

        # Step 3: Plan
        if verbose:
            print("[3/4] Creating edit plan...", flush=True)

        if use_content_detection:
            from .detect import create_content_plan
            keep_map = {
                "songs_only": ["music", "singing", "speech_over_music"],
                "speech_only": ["speech", "talking_over_music", "speech_over_music"],
                "no_silence": ["speech", "music", "singing", "talking_over_music", "speech_over_music"],
            }
            plan, blocks = create_content_plan(
                input_path, transcript, probe.duration,
                keep_types=keep_map[preset],
                detector=detector,
                verbose=verbose,
            )
            if save_artifacts:
                blocks_data = [
                    {"start": b.start, "end": b.end, "duration": b.duration,
                     "type": b.content_type, "rms_db": round(b.mean_volume, 1),
                     "crest_factor": round(b.crest_factor, 1),
                     "dynamic_range": round(b.dynamic_range, 1),
                     "zero_crossing_rate": round(b.zero_crossing_rate, 4),
                     "confidence": round(b.confidence, 2)}
                    for b in blocks
                ]
                blocks_path = artifacts_dir / "content_blocks.json"
                artifacts_dir.mkdir(parents=True, exist_ok=True)  # ensure dir exists
                with open(blocks_path, "w") as f:
                    json.dump(blocks_data, f, indent=2)
                artifacts["content_blocks"] = str(blocks_path)
        else:
            editor = HeuristicEditor()
            plan = editor.create_plan(
                transcript, probe.duration,
                remove_fillers=remove_fillers,
                remove_repetitions=remove_repetitions,
                remove_silence=remove_silence,
                min_silence=min_silence,
            )

        if save_artifacts:
            plan_path = artifacts_dir / "plan.json"
            with open(plan_path, "w") as f:
                json.dump(plan.to_dict(), f, indent=2)
            artifacts["plan"] = str(plan_path)

        if verbose:
            print(f"    Original: {plan.original_duration:.1f}s", flush=True)
            print(f"    Edited:   {plan.edited_duration:.1f}s", flush=True)
            print(f"    Removed:  {plan.removed_duration:.1f}s ({plan.removed_duration / plan.original_duration * 100:.1f}%)", flush=True)

        # Step 4: Render video
        if verbose:
            print(f"[4/4] Rendering: {output_path}", flush=True)
        renderer = FFmpegVideoRenderer()
        renderer.render(input_path, output_path, plan, copy_codec=copy_codec, verbose=verbose)
        artifacts["output"] = output_path

        if verbose:
            print(f"\n✓ Done! Output: {output_path}", flush=True)

        return EditResult(
            input_path=input_path,
            output_path=output_path,
            probe=probe,
            transcript=transcript,
            plan=plan,
            success=True,
            artifacts=artifacts,
        )

    except Exception as e:
        return EditResult(
            input_path=input_path,
            output_path=output_path,
            probe=probe,
            transcript=transcript,
            plan=plan,
            success=False,
            error=str(e),
            artifacts=artifacts,
        )
