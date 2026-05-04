"""CLI entry point for praisonai-editor.

Usage:
    praisonai-editor edit input.mp3 --output edited.mp3
    praisonai-editor transcribe input.mp3 --format srt
    praisonai-editor convert input.mp4 --format mp3
    praisonai-editor probe input.mp3
    praisonai-editor trim talk.mp3 --start "..." --end "..." --verify --verify-tail-forbid "..."
    praisonai-editor eval trimmed.mp3 --head-contains "..." --tail-forbid "..."
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="praisonai-editor",
        description="AI-powered audio & video editor — transcribe, clean, and edit media",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- probe ---
    probe_parser = subparsers.add_parser("probe", help="Probe media file metadata")
    probe_parser.add_argument("input", help="Input media file")
    probe_parser.add_argument("--output", "-o", help="Output JSON file")
    probe_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # --- convert ---
    convert_parser = subparsers.add_parser("convert", help="Convert media format")
    convert_parser.add_argument("input", help="Input media file")
    convert_parser.add_argument("--output", "-o", help="Output file path")
    convert_parser.add_argument("--format", "-f", default="mp3", choices=["mp3", "wav", "m4a"],
                                help="Output format (default: mp3)")
    convert_parser.add_argument("--bitrate", "-b", default="192k", help="Audio bitrate")

    # --- transcribe ---
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe audio/video")
    trans_parser.add_argument("input", help="Input media file")
    trans_parser.add_argument("--output", "-o", help="Output file")
    trans_parser.add_argument("--format", "-f", choices=["txt", "srt", "json"], default="srt")
    trans_parser.add_argument("--local", action="store_true", help="Use local faster-whisper")
    trans_parser.add_argument("--language", help="Language code (e.g., en)")
    trans_parser.add_argument(
        "--model",
        "-m",
        help="Model id: whisper-1 for API (default); tiny, base, small, … for --local",
    )

    # --- trim (phrase boundaries) ---
    trim_parser = subparsers.add_parser(
        "trim",
        help="Transcribe then cut by phrase markers (ffmpeg stream copy)",
    )
    trim_parser.add_argument("input", help="Input media file")
    trim_parser.add_argument("--output", "-o", help="Output path (default: *_trimmed.ext)")
    trim_parser.add_argument(
        "--start",
        required=True,
        help="First words to keep (inclusive); fuzzy match on transcript",
    )
    trim_parser.add_argument(
        "--end",
        required=True,
        help="Cut before this phrase (exclusive); phrase omitted from output",
    )
    trim_parser.add_argument(
        "--end-first",
        action="store_true",
        help="Match first occurrence of --end instead of last",
    )
    trim_parser.add_argument(
        "--end-guard",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "Subtract SEC from the exclusive end time after phrase detection (default 0). "
            "Use a small value (e.g. 0.2–0.5) when speech is still audible just before the "
            "end phrase because word timestamps start slightly late"
        ),
    )
    trim_parser.add_argument(
        "--trim-boundaries",
        choices=("phrase-first", "window"),
        default="window",
        help=(
            "phrase-first: clip starts at the first word of --start (inclusive) and ends "
            "before the first word of --end (exclusive). window (default): legacy sliding-window match"
        ),
    )
    trim_parser.add_argument(
        "--local",
        action="store_true",
        help="Use faster-whisper locally instead of default OpenAI whisper-1",
    )
    trim_parser.add_argument("--language", help="Language code (e.g., en)")
    trim_parser.add_argument(
        "--model",
        "-m",
        help="API: whisper-1 default. Local: base default; use tiny for speed",
    )
    trim_parser.add_argument(
        "--transcript",
        "-T",
        metavar="FILE",
        help="Use this transcript JSON instead of running ASR",
    )
    trim_parser.add_argument(
        "--force-transcribe",
        "--no-cache",
        action="store_true",
        dest="force_transcribe",
        help=(
            "Do not load transcript cache (~/.praisonai/editor/… or legacy sidecar); "
            "run ASR; on success replace transcript.json there (unless --no-cache-write)"
        ),
    )
    trim_parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="After ASR, do not write ~/.praisonai/editor/{stem}_{hash}/transcript.json",
    )
    trim_parser.add_argument(
        "--refine-openai",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After fuzzy phrase match, call OpenAI Chat Completions (gpt-4o-mini by default; "
            "not PraisonAI agents) to adjust start/end from transcript text. "
            "Requires OPENAI_API_KEY; use --no-refine-openai to skip"
        ),
    )
    trim_parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After each crop, run eval on the output; if checks fail, increase --end-guard by "
            "--verify-end-guard-step and re-trim (up to --verify-max-retries attempts)"
        ),
    )
    trim_parser.add_argument(
        "--verify-head-contains",
        metavar="TEXT",
        help="With --verify: substring that must appear in the opening (normalised); pass to eval",
    )
    trim_parser.add_argument(
        "--verify-tail-forbid",
        action="append",
        default=[],
        metavar="TEXT",
        help="With --verify: substring that must not appear near the end (repeatable); pass to eval",
    )
    trim_parser.add_argument(
        "--verify-max-retries",
        type=int,
        default=3,
        metavar="N",
        help="With --verify: maximum trim+eval attempts (default 3)",
    )
    trim_parser.add_argument(
        "--verify-end-guard-step",
        type=float,
        default=0.15,
        metavar="SEC",
        help="With --verify: add SEC to --end-guard before each retry after a failed eval (default 0.15)",
    )
    trim_parser.add_argument(
        "--verify-quick",
        action="store_true",
        help="With --verify: use shorter eval windows (same as eval --quick)",
    )
    trim_parser.add_argument(
        "--verify-ai-judge",
        action="store_true",
        help=(
            "With --verify: ask OpenAI (chat JSON) if opening/closing transcripts match intent; "
            "combine with --verify-head-contains / --verify-tail-forbid for derived intent"
        ),
    )
    trim_parser.add_argument(
        "--verify-ai-start-intent",
        metavar="TEXT",
        default=None,
        help="With --verify-ai-judge: natural-language intent for how the clip should open",
    )
    trim_parser.add_argument(
        "--verify-ai-end-intent",
        metavar="TEXT",
        default=None,
        help="With --verify-ai-judge: natural-language intent for how the clip should end",
    )
    trim_parser.add_argument(
        "--verify-ai-judge-model",
        metavar="MODEL",
        default=None,
        help="With --verify-ai-judge: chat model (default OPENAI_EVAL_JUDGE_MODEL or gpt-4o-mini)",
    )
    trim_parser.add_argument(
        "--verify-no-word-timings",
        action="store_true",
        help="With --verify: omit opening/closing word-level times from the eval report",
    )
    trim_parser.add_argument(
        "--verify-word-timing-limit",
        type=int,
        default=40,
        metavar="N",
        help="With --verify: max words to keep in opening_words_timed / closing_words_timed (default 40)",
    )
    trim_parser.add_argument(
        "--verify-quiet",
        action="store_true",
        help="With --verify: do not print timed word previews to stderr",
    )

    # --- eval (trim verification) ---
    from .trim_eval import (
        DEFAULT_HEAD_CONTEXT_AFTER_SEC,
        DEFAULT_HEAD_CONTEXT_BEFORE_SEC,
        DEFAULT_HEAD_WINDOW_SEC,
        DEFAULT_TAIL_CONTEXT_AFTER_SEC,
        DEFAULT_TAIL_CONTEXT_BEFORE_SEC,
        DEFAULT_TAIL_WINDOW_SEC,
        QUICK_HEAD_CONTEXT_AFTER_SEC,
        QUICK_HEAD_CONTEXT_BEFORE_SEC,
        QUICK_HEAD_WINDOW_SEC,
        QUICK_TAIL_CONTEXT_AFTER_SEC,
        QUICK_TAIL_CONTEXT_BEFORE_SEC,
        QUICK_TAIL_WINDOW_SEC,
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Verify generated/trimmed audio: transcribe first & last few seconds (+ context bands)",
    )
    eval_parser.add_argument("input", help="Media file (e.g. trim output)")
    eval_parser.add_argument(
        "--quick",
        action="store_true",
        help="Use shorter samples (5s first/last, tighter context) for a fast check",
    )
    eval_parser.add_argument(
        "--head-sec",
        type=float,
        default=DEFAULT_HEAD_WINDOW_SEC,
        help=f"First N seconds of the file to transcribe (default {DEFAULT_HEAD_WINDOW_SEC:g})",
    )
    eval_parser.add_argument(
        "--tail-sec",
        type=float,
        default=DEFAULT_TAIL_WINDOW_SEC,
        help=f"Last N seconds of the file to transcribe (default {DEFAULT_TAIL_WINDOW_SEC:g})",
    )
    eval_parser.add_argument(
        "--head-before-sec",
        type=float,
        default=DEFAULT_HEAD_CONTEXT_BEFORE_SEC,
        metavar="SEC",
        help=(
            "Transcribe last SEC seconds inside the head window (before head/after boundary; 0=off; "
            f"default {DEFAULT_HEAD_CONTEXT_BEFORE_SEC:g})"
        ),
    )
    eval_parser.add_argument(
        "--head-after-sec",
        type=float,
        default=DEFAULT_HEAD_CONTEXT_AFTER_SEC,
        metavar="SEC",
        help=(
            "Transcribe SEC seconds after the head window (0=off; "
            f"default {DEFAULT_HEAD_CONTEXT_AFTER_SEC:g})"
        ),
    )
    eval_parser.add_argument(
        "--tail-before-sec",
        type=float,
        default=DEFAULT_TAIL_CONTEXT_BEFORE_SEC,
        metavar="SEC",
        help=(
            "Transcribe SEC seconds before the tail window (0=off; "
            f"default {DEFAULT_TAIL_CONTEXT_BEFORE_SEC:g})"
        ),
    )
    eval_parser.add_argument(
        "--tail-after-sec",
        type=float,
        default=DEFAULT_TAIL_CONTEXT_AFTER_SEC,
        metavar="SEC",
        help=(
            "Transcribe last SEC seconds of the file for closing words (0=off; "
            f"default {DEFAULT_TAIL_CONTEXT_AFTER_SEC:g})"
        ),
    )
    eval_parser.add_argument(
        "--head-contains",
        metavar="TEXT",
        help="Substring that must appear in head region (after normalisation), including context",
    )
    eval_parser.add_argument(
        "--tail-contains",
        metavar="TEXT",
        help="Substring that must appear in tail region (after normalisation), including context",
    )
    eval_parser.add_argument(
        "--tail-forbid",
        action="append",
        default=[],
        metavar="TEXT",
        help="Substring that must not appear in tail region (repeatable); checked with before/core/after merged",
    )
    eval_parser.add_argument("--local", action="store_true", help="Use faster-whisper locally")
    eval_parser.add_argument("--language", help="Language code (e.g. en)")
    eval_parser.add_argument("-m", "--model", help="Transcription model")
    eval_parser.add_argument("-o", "--output", help="Write JSON report to this path")
    eval_parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    eval_parser.add_argument(
        "--force-transcribe",
        "--no-cache",
        action="store_true",
        dest="force_transcribe",
        help="Ignore eval transcript cache (~/.praisonai/editor/eval/…) and re-transcribe each segment",
    )
    eval_parser.add_argument(
        "--no-cache-write",
        action="store_true",
        help="Do not write eval transcript cache after ASR",
    )
    eval_parser.add_argument(
        "--ai-judge",
        action="store_true",
        help=(
            "After substring checks, call OpenAI (chat JSON) on opening/closing transcripts; "
            "requires OPENAI_API_KEY; intents from --ai-start-intent/--ai-end-intent or derived from "
            "--head-contains / --tail-forbid"
        ),
    )
    eval_parser.add_argument(
        "--ai-start-intent",
        metavar="TEXT",
        default=None,
        help="Natural-language intent for how the trimmed clip should open",
    )
    eval_parser.add_argument(
        "--ai-end-intent",
        metavar="TEXT",
        default=None,
        help="Natural-language intent for how the trimmed clip should end (e.g. end before prayer)",
    )
    eval_parser.add_argument(
        "--ai-judge-model",
        metavar="MODEL",
        default=None,
        help="Chat model for AI judge (default OPENAI_EVAL_JUDGE_MODEL or gpt-4o-mini)",
    )
    eval_parser.add_argument(
        "--no-word-timings",
        action="store_true",
        help="Omit opening_words_timed / closing_words_timed (OpenAI word timestamps by default)",
    )
    eval_parser.add_argument(
        "--word-timing-limit",
        type=int,
        default=40,
        metavar="N",
        help="Max words in opening/closing timed samples (default 40)",
    )
    eval_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Do not print timed word previews",
    )

    # --- plan ---
    plan_parser = subparsers.add_parser("plan", help="Create edit plan from transcript")
    plan_parser.add_argument("input", help="Input media file")
    plan_parser.add_argument("--output", "-o", help="Output JSON file")
    plan_parser.add_argument("--preset", "-p", default="podcast",
                             choices=["podcast", "meeting", "course", "clean"])
    plan_parser.add_argument("--local", action="store_true", help="Use local whisper")

    # --- edit ---
    edit_parser = subparsers.add_parser("edit", help="Full editing pipeline")
    edit_parser.add_argument("input", help="Input media file")
    edit_parser.add_argument("--output", "-o", help="Output file")
    edit_parser.add_argument("--preset", "-p", default="podcast",
                             choices=["podcast", "meeting", "course", "clean",
                                      "songs_only", "speech_only", "no_silence"])
    edit_parser.add_argument("--prompt", help="Natural language editing instructions (uses AI agent)")
    edit_parser.add_argument("--no-fillers", action="store_true", help="Keep filler words")
    edit_parser.add_argument("--no-repetitions", action="store_true", help="Keep repetitions")
    edit_parser.add_argument("--no-silence", action="store_true", help="Keep silences")
    edit_parser.add_argument("--local", action="store_true", help="Use local whisper")
    edit_parser.add_argument("--language",        help="Language code for transcription (e.g., 'en', 'es')"
    )
    edit_parser.add_argument(
        "--detector",
        choices=["auto", "ensemble", "ina", "librosa", "ffmpeg"],
        default="auto",
        help="Audio content detector to use (default: auto -> ensemble), ina (CNN), librosa (spectral), ffmpeg (heuristic)"
    )
    edit_parser.add_argument("--reencode", action="store_true", help="Re-encode instead of copy")
    edit_parser.add_argument(
        "--demix",
        action="store_true",
        default=False,
        help=(
            "Use Demucs stem separation to distinguish singing from talking over music. "
            "Requires: pip install praisonai-editor[demix]"
        ),
    )
    edit_parser.add_argument(
        "--primary-zone",
        action="store_true",
        default=False,
        dest="primary_zone_only",
        help=(
            "Auto-detect and keep only the primary (largest) singing zone. "
            "Trims any scatter singing before/after the main performance automatically."
        ),
    )
    edit_parser.add_argument("--verbose", "-v", action="store_true")
    edit_parser.add_argument("--no-artifacts", action="store_true", help="Don't save artifacts")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "probe":
            return cmd_probe(args)
        elif args.command == "convert":
            return cmd_convert(args)
        elif args.command == "transcribe":
            return cmd_transcribe(args)
        elif args.command == "plan":
            return cmd_plan(args)
        elif args.command == "edit":
            return cmd_edit(args)
        elif args.command == "trim":
            return cmd_trim(args)
        elif args.command == "eval":
            return cmd_eval(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_probe(args):
    from .probe import probe_media

    result = probe_media(args.input)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved to: {args.output}")
    elif args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"File: {result.path}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Type: {'Video + Audio' if result.has_video else 'Audio only'}")
        if result.has_video:
            print(f"Resolution: {result.width}x{result.height}")
            print(f"FPS: {result.fps:.2f}")
            print(f"Video codec: {result.video_codec}")
        if result.audio_codec:
            print(f"Audio codec: {result.audio_codec}")
            print(f"Audio: {result.audio_sample_rate}Hz, {result.audio_channels}ch")
        print(f"Size: {result.size_bytes / 1024 / 1024:.2f} MB")
    return 0


def cmd_convert(args):
    from .convert import convert_media

    output = args.output
    if not output:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}.{args.format}")

    result = convert_media(args.input, output, bitrate=args.bitrate)
    print(f"✓ Converted: {result}")
    return 0


def cmd_transcribe(args):
    from .transcribe import transcribe_audio

    result = transcribe_audio(
        args.input,
        use_local=args.local,
        language=args.language,
        model=args.model,
    )

    output_format = args.format
    if args.output:
        ext = Path(args.output).suffix.lower()
        if ext in [".txt", ".srt", ".json"]:
            output_format = ext[1:]

    if output_format == "txt":
        content = result.text
    elif output_format == "srt":
        content = result.to_srt()
    else:
        content = json.dumps(result.to_dict(), indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(content)
        print(f"Saved to: {args.output}")
    else:
        print(content)
    return 0


def cmd_trim(args):
    from .phrase_trim import trim_between_phrase_markers
    from .trim_eval import (
        DEFAULT_HEAD_CONTEXT_AFTER_SEC,
        DEFAULT_HEAD_CONTEXT_BEFORE_SEC,
        DEFAULT_HEAD_WINDOW_SEC,
        DEFAULT_TAIL_CONTEXT_AFTER_SEC,
        DEFAULT_TAIL_CONTEXT_BEFORE_SEC,
        DEFAULT_TAIL_WINDOW_SEC,
        QUICK_HEAD_CONTEXT_AFTER_SEC,
        QUICK_HEAD_CONTEXT_BEFORE_SEC,
        QUICK_HEAD_WINDOW_SEC,
        QUICK_TAIL_CONTEXT_AFTER_SEC,
        QUICK_TAIL_CONTEXT_BEFORE_SEC,
        QUICK_TAIL_WINDOW_SEC,
        evaluate_trim_edges,
    )

    output = args.output
    if not output:
        p = Path(args.input)
        output = str(p.parent / f"{p.stem}_trimmed{p.suffix}")

    end_guard = float(args.end_guard)
    max_attempts = max(1, int(args.verify_max_retries)) if args.verify else 1

    for attempt in range(max_attempts):
        path = trim_between_phrase_markers(
            args.input,
            output,
            start_phrase=args.start,
            end_phrase=args.end,
            end_last_match=not args.end_first,
            use_local=args.local,
            language=args.language,
            model=args.model,
            transcript_path=args.transcript,
            use_transcript_cache=True,
            write_transcript_cache=not args.no_cache_write,
            force_transcribe=args.force_transcribe,
            refine_with_openai=args.refine_openai,
            end_guard_seconds=end_guard,
            trim_boundaries=args.trim_boundaries,
        )
        print(f"✓ Wrote: {path}", flush=True)

        if not args.verify:
            return 0

        vf = args.verify_tail_forbid if args.verify_tail_forbid else None
        if args.verify_quick:
            hs, ts = QUICK_HEAD_WINDOW_SEC, QUICK_TAIL_WINDOW_SEC
            hb, ha = QUICK_HEAD_CONTEXT_BEFORE_SEC, QUICK_HEAD_CONTEXT_AFTER_SEC
            tb, ta = QUICK_TAIL_CONTEXT_BEFORE_SEC, QUICK_TAIL_CONTEXT_AFTER_SEC
        else:
            hs, ts = DEFAULT_HEAD_WINDOW_SEC, DEFAULT_TAIL_WINDOW_SEC
            hb, ha = DEFAULT_HEAD_CONTEXT_BEFORE_SEC, DEFAULT_HEAD_CONTEXT_AFTER_SEC
            tb, ta = DEFAULT_TAIL_CONTEXT_BEFORE_SEC, DEFAULT_TAIL_CONTEXT_AFTER_SEC

        rep = evaluate_trim_edges(
            path,
            head_window_sec=hs,
            tail_window_sec=ts,
            head_context_before_sec=hb,
            head_context_after_sec=ha,
            tail_context_before_sec=tb,
            tail_context_after_sec=ta,
            head_contains=args.verify_head_contains,
            tail_forbid=vf,
            language=args.language,
            use_local=args.local,
            model=args.model,
            ai_judge=bool(args.verify_ai_judge),
            ai_start_intent=args.verify_ai_start_intent,
            ai_end_intent=args.verify_ai_end_intent,
            ai_judge_model=args.verify_ai_judge_model,
            include_word_timings=not args.verify_no_word_timings,
            word_timing_limit=args.verify_word_timing_limit,
            quiet=args.verify_quiet,
        )
        if rep.ok:
            print("✓ Verification passed (eval)", flush=True)
            return 0

        print(f"Verification failed (attempt {attempt + 1}/{max_attempts}): {rep.failures}", flush=True)
        if attempt >= max_attempts - 1:
            return 1
        end_guard = end_guard + float(args.verify_end_guard_step)
        print(f"Re-trimming with end_guard={end_guard:g}s", flush=True)


def cmd_eval(args):
    from .trim_eval import (
        evaluate_trim_edges,
        QUICK_HEAD_CONTEXT_AFTER_SEC,
        QUICK_HEAD_CONTEXT_BEFORE_SEC,
        QUICK_HEAD_WINDOW_SEC,
        QUICK_TAIL_CONTEXT_AFTER_SEC,
        QUICK_TAIL_CONTEXT_BEFORE_SEC,
        QUICK_TAIL_WINDOW_SEC,
    )

    tail_forbid = args.tail_forbid if getattr(args, "tail_forbid", None) else None
    if args.quick:
        hs, ts = QUICK_HEAD_WINDOW_SEC, QUICK_TAIL_WINDOW_SEC
        hb, ha = QUICK_HEAD_CONTEXT_BEFORE_SEC, QUICK_HEAD_CONTEXT_AFTER_SEC
        tb, ta = QUICK_TAIL_CONTEXT_BEFORE_SEC, QUICK_TAIL_CONTEXT_AFTER_SEC
    else:
        hs, ts = args.head_sec, args.tail_sec
        hb, ha = args.head_before_sec, args.head_after_sec
        tb, ta = args.tail_before_sec, args.tail_after_sec

    r = evaluate_trim_edges(
        args.input,
        head_window_sec=hs,
        tail_window_sec=ts,
        head_context_before_sec=hb,
        head_context_after_sec=ha,
        tail_context_before_sec=tb,
        tail_context_after_sec=ta,
        head_contains=args.head_contains,
        tail_contains=args.tail_contains,
        tail_forbid=tail_forbid,
        language=args.language,
        use_local=args.local,
        model=args.model,
        use_eval_cache=not args.force_transcribe,
        write_eval_cache=not args.no_cache_write,
        force_transcribe=args.force_transcribe,
        ai_judge=args.ai_judge,
        ai_start_intent=args.ai_start_intent,
        ai_end_intent=args.ai_end_intent,
        ai_judge_model=args.ai_judge_model,
        include_word_timings=not args.no_word_timings,
        word_timing_limit=args.word_timing_limit,
        quiet=args.quiet,
    )
    payload = r.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved to: {args.output}")
    if args.json or not args.output:
        print(json.dumps(payload, indent=2))
    elif args.output:
        print(
            f"ok={r.ok} asr={r.asr_backend}/{r.asr_model} "
            f"opening_words={len(r.opening_words_timed)} closing_words={len(r.closing_words_timed)} "
            f"failures={r.failures}"
        )
    return 0 if r.ok else 1


def cmd_plan(args):
    from .probe import probe_media
    from .transcribe import transcribe_audio
    from .plan import create_edit_plan

    print(f"Probing: {args.input}")
    probe = probe_media(args.input)

    print(f"Transcribing ({probe.duration:.1f}s)...")
    transcript = transcribe_audio(args.input, use_local=args.local)

    print("Creating edit plan...")
    plan = create_edit_plan(transcript, probe.duration, preset=args.preset)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        print(f"Saved to: {args.output}")
    else:
        print(json.dumps(plan.to_dict(), indent=2))

    print(f"\nOriginal: {plan.original_duration:.1f}s")
    print(f"Edited:   {plan.edited_duration:.1f}s")
    print(f"Removed:  {plan.removed_duration:.1f}s ({plan.removed_duration / plan.original_duration * 100:.1f}%)")
    return 0


def cmd_edit(args):
    if args.prompt:
        # Prompt-based editing via agent
        from .agent_pipeline import prompt_edit

        result = prompt_edit(
            args.input,
            args.prompt,
            output_path=args.output,
            use_local_whisper=args.local,
            verbose=args.verbose,
        )
    else:
        # Standard pipeline editing
        from .pipeline import edit_media

        result = edit_media(
            args.input,
            output_path=args.output,
            preset=args.preset,
            detector=args.detector,
            demix=args.demix,
            primary_zone_only=args.primary_zone_only,
            remove_fillers=not args.no_fillers,
            remove_repetitions=not args.no_repetitions,
            remove_silence=not args.no_silence,
            use_local_whisper=args.local,
            language=args.language,
            copy_codec=not args.reencode,
            verbose=args.verbose,
            save_artifacts=not args.no_artifacts,
        )

    if result.success:
        print(f"\n✓ Success! Output: {result.output_path}")
        if result.artifacts:
            print("\nArtifacts:")
            for name, path in result.artifacts.items():
                print(f"  {name}: {path}")
        return 0
    else:
        print(f"\n✗ Failed: {result.error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
