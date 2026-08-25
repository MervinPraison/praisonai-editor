"""CLI entry point for praisonai-editor.

Usage:
    praisonai-editor edit input.mp3 --output edited.mp3
    praisonai-editor transcribe input.mp3 --format srt
    praisonai-editor extract-text transcript.json -o transcript.txt
    praisonai-editor convert input.mp4 --format mp3
    praisonai-editor concat part1.m4a part2.m4a -o joined.m4a
    praisonai-editor conform mastered.wav --duration 3540.2
    praisonai-editor normalize input.m4a --in-place
    praisonai-editor master input.m4a --preset speech
    praisonai-editor denoise input.m4a --noise-reduction 20
    praisonai-editor probe input.mp3
    praisonai-editor trim talk.mp3 --start "..." --end "..." --verify --verify-tail-forbid "..."
    praisonai-editor remove talk.mp3 --range 11:53-12:43
    praisonai-editor eval trimmed.mp3 --head-contains "..." --tail-forbid "..."
    praisonai-editor demix talk.mp3 --vocals-output vocals.wav --instruments-output inst.wav
    praisonai-editor session start talk.mp3
    praisonai-editor session undo <session-id>
    praisonai-editor session redo <session-id>
    praisonai-editor session jump <session-id> -1
    praisonai-editor session history <session-id>
    praisonai-editor session reset <session-id>
    praisonai-editor session end <session-id>
    praisonai-editor session prune --max-age-seconds 604800
    praisonai-editor apply plan.yaml
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
    convert_parser.add_argument(
        "--format", "-f", default="mp3",
        choices=["mp3", "wav", "m4a", "aac", "ogg", "flac"],
        help="Output format (default: mp3)",
    )
    convert_parser.add_argument("--bitrate", "-b", default="192k", help="Audio bitrate")

    # --- concat ---
    concat_parser = subparsers.add_parser(
        "concat",
        help="Concatenate audio files (stream copy, or --reencode for mixed inputs)",
    )
    concat_parser.add_argument("inputs", nargs="+", help="Input audio files, in order")
    concat_parser.add_argument("--output", "-o", required=True, help="Output file path")
    concat_parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode via concat filter (needed when inputs have differing codecs/rates)",
    )
    concat_parser.add_argument("--bitrate", "-b", default="192k", help="AAC bitrate when re-encoding")
    concat_parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        metavar="HZ",
        help="Target sample rate when re-encoding (default 48000)",
    )
    concat_parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=[1, 2],
        help="Target channels when re-encoding: 1 mono, 2 stereo (default 2)",
    )
    concat_parser.add_argument("--verbose", "-v", action="store_true")
    concat_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- conform ---
    conform_parser = subparsers.add_parser(
        "conform",
        help="Conform mastered audio for splicing (resample, channel layout, exact length)",
    )
    conform_parser.add_argument("input", help="Input audio file")
    conform_parser.add_argument("--output", "-o", help="Output file (default: {stem}_conformed.m4a)")
    conform_parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        metavar="HZ",
        help="Target sample rate (default 48000)",
    )
    conform_parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=(1, 2),
        help="Target channels: 1 mono, 2 stereo (default 2)",
    )
    conform_parser.add_argument("--bitrate", "-b", default="192k", help="AAC bitrate")
    conform_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECS",
        help="Force EXACT output length (trim if longer, pad silence if shorter)",
    )
    conform_parser.add_argument("--verbose", "-v", action="store_true")
    conform_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- demix ---
    demix_parser = subparsers.add_parser(
        "demix",
        help="Isolate vocals from instruments (Demucs stem separation)",
    )
    demix_parser.add_argument("input", help="Input media file")
    demix_parser.add_argument(
        "--vocals-output",
        dest="vocals_output",
        help="Output path for the vocals stem (default: {stem}.vocals.wav)",
    )
    demix_parser.add_argument(
        "--instruments-output",
        dest="instruments_output",
        help="Output path for the instruments stem (default: {stem}.instruments.wav)",
    )
    demix_parser.add_argument(
        "--model",
        default="mdx_extra",
        help="Demucs model name (default: mdx_extra)",
    )
    demix_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu (default, safe everywhere) or mps (Apple Silicon GPU)",
    )
    demix_parser.add_argument("--verbose", "-v", action="store_true")
    demix_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- normalize ---
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Optimise quiet audio loudness (volumedetect → loudnorm when needed)",
    )
    norm_parser.add_argument("input", help="Input audio file")
    norm_parser.add_argument("--output", "-o", help="Output file (default: input.norm.m4a)")
    norm_parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input when normalisation runs (no-op copy path if already loud enough)",
    )
    norm_parser.add_argument(
        "--force",
        action="store_true",
        help="Always apply loudnorm even when volume is already OK",
    )
    norm_parser.add_argument(
        "--mean-threshold",
        type=float,
        default=-22.0,
        metavar="DB",
        help="Normalise when mean_volume below this (default -22)",
    )
    norm_parser.add_argument(
        "--max-threshold",
        type=float,
        default=-8.0,
        metavar="DB",
        help="Normalise when max_volume below this (default -8)",
    )
    norm_parser.add_argument(
        "--target-lufs",
        type=float,
        default=-16.0,
        metavar="LUFS",
        help="Integrated loudness target (default -16)",
    )
    norm_parser.add_argument(
        "--true-peak",
        type=float,
        default=-1.5,
        metavar="DBTP",
        help="True-peak ceiling in dBTP (default -1.5)",
    )
    norm_parser.add_argument(
        "--lra",
        type=float,
        default=11.0,
        metavar="LU",
        help="Loudness-range target (default 11)",
    )
    norm_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- master ---
    master_parser = subparsers.add_parser(
        "master",
        help="Master audio to a streaming loudness target (two-pass EBU R128 loudnorm)",
    )
    master_parser.add_argument("input", help="Input audio file")
    master_parser.add_argument("--output", "-o", help="Output file (default: {stem}.mastered.m4a)")
    master_parser.add_argument(
        "--preset",
        "-p",
        default="speech",
        choices=("speech", "music", "auto"),
        help="Mastering preset; auto picks speech vs music from measured stats (default speech)",
    )
    master_parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        metavar="LUFS",
        help="Integrated loudness target (default -14, YouTube norm)",
    )
    master_parser.add_argument(
        "--true-peak",
        type=float,
        default=-1.5,
        metavar="DBTP",
        help="True-peak ceiling in dBTP (default -1.5)",
    )
    master_parser.add_argument(
        "--lra",
        type=float,
        default=None,
        metavar="LU",
        help="Loudness-range target (default: preset-driven — speech 11, music 15)",
    )
    master_parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        metavar="HZ",
        help="Output sample rate (default 48000)",
    )
    master_parser.add_argument(
        "--channels",
        type=int,
        default=2,
        choices=(1, 2),
        help="Output channels: 1 mono, 2 stereo (default 2)",
    )
    master_parser.add_argument("--bitrate", "-b", default="192k", help="AAC bitrate")
    master_parser.add_argument(
        "--chain",
        action="append",
        metavar="FILTER",
        help=(
            "One ffmpeg -af filter expression (e.g. 'acompressor=threshold=-18dB:ratio=3'); "
            "repeat --chain for each filter. Fully REPLACES the preset's own pre-chain "
            "(loudnorm + limiter + resample always still run after it). "
            "Default: preset-driven pre-chain."
        ),
    )
    master_parser.add_argument("--verbose", "-v", action="store_true")
    master_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- denoise ---
    denoise_parser = subparsers.add_parser(
        "denoise",
        help="Reduce background noise/hiss with ffmpeg's FFT denoiser (afftdn)",
    )
    denoise_parser.add_argument("input", help="Input audio file")
    denoise_parser.add_argument(
        "--output", "-o", help="Output file (default: {stem}_denoised.m4a)"
    )
    denoise_parser.add_argument(
        "--noise-reduction",
        type=float,
        default=12.0,
        metavar="DB",
        help="Amount of noise reduction, 0.01-97 (default 12)",
    )
    denoise_parser.add_argument(
        "--noise-floor",
        type=float,
        default=-50.0,
        metavar="DB",
        help="Expected noise floor, -80 to -20 (default -50)",
    )
    denoise_parser.add_argument(
        "--no-track-noise",
        action="store_true",
        help="Disable adapting to noise that changes over the file (default: adapts)",
    )
    denoise_parser.add_argument("--bitrate", "-b", default="192k", help="AAC bitrate")
    denoise_parser.add_argument("--verbose", "-v", action="store_true")
    denoise_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- word-gaps ---
    gaps_parser = subparsers.add_parser(
        "word-gaps",
        help="Shorten long pauses between words to a target length (keeps some, unlike remove)",
    )
    gaps_parser.add_argument("input", help="Input audio file")
    gaps_parser.add_argument("--output", "-o", help="Output file (default: {stem}_cut{ext})")
    gaps_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Only gaps longer than this are touched (default 0.5)",
    )
    gaps_parser.add_argument(
        "--target",
        type=float,
        default=0.25,
        metavar="SEC",
        help="Shorten each qualifying gap TO this length, must be < --threshold (default 0.25)",
    )
    gaps_parser.add_argument(
        "--transcript",
        "-T",
        metavar="FILE",
        help="Use this transcript JSON instead of running ASR",
    )
    gaps_parser.add_argument(
        "--local", action="store_true", help="Transcribe with local faster-whisper (only used without --transcript)"
    )
    gaps_parser.add_argument("--language", help="Language code (e.g., en) (only used without --transcript)")
    gaps_parser.add_argument("--model", "-m", help="ASR model id (only used without --transcript)")
    gaps_parser.add_argument(
        "--reencode", action="store_true", help="Re-encode instead of stream-copy (slower, frame-accurate)"
    )
    gaps_parser.add_argument("--verbose", "-v", action="store_true")
    gaps_parser.add_argument("--json", action="store_true", help="Print result as JSON")

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
    trans_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Speed audio before ASR (e.g. 2.0 halves API cost; timestamps scaled back)",
    )
    trans_parser.add_argument(
        "--vad-filter",
        action="store_true",
        help=(
            "Local-only (--local): filter out non-speech before transcription for more "
            "accurate word gap/silence timestamps. Silently ignored for the OpenAI API path."
        ),
    )

    # --- extract-text (from transcript JSON) ---
    extract_parser = subparsers.add_parser(
        "extract-text",
        help="Extract plain text from a transcript JSON file",
    )
    extract_parser.add_argument("input", help="Transcript JSON file")
    extract_parser.add_argument(
        "--output",
        "-o",
        help="Output .txt path (default: same stem as input, .txt extension)",
    )

    # --- remove (time ranges) ---
    remove_parser = subparsers.add_parser(
        "remove",
        help="Remove time ranges from audio/video (e.g. 11:53-12:43)",
    )
    remove_parser.add_argument("input", help="Input media file")
    remove_parser.add_argument("--output", "-o", help="Output path (default: *_cut.ext)")
    remove_parser.add_argument(
        "--range",
        "-r",
        action="append",
        metavar="START-END",
        help="Range to remove, e.g. 11:53-12:43 (repeatable)",
    )
    remove_parser.add_argument(
        "--from",
        dest="cut_from",
        metavar="TIME",
        help="Single range start (use with --to)",
    )
    remove_parser.add_argument(
        "--to",
        dest="cut_to",
        metavar="TIME",
        help="Single range end (use with --from)",
    )
    remove_parser.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode instead of stream copy (slower, cleaner cuts)",
    )
    remove_parser.add_argument(
        "--transcript",
        "-T",
        metavar="FILE",
        help=(
            "A transcript JSON synced to this file -- enables both re-timing "
            "the transcript to match the cut output, and word-boundary "
            "refinement (see --no-refine-boundaries)"
        ),
    )
    remove_parser.add_argument(
        "--no-refine-boundaries",
        action="store_true",
        help=(
            "With --transcript: cut at the raw reported timestamps instead of "
            "nudging each edge to the nearest real acoustic gap first (default: "
            "refine -- an ASR timestamp is a coarse hint, not a precise boundary, "
            "especially for fast/connected speech)"
        ),
    )
    remove_parser.add_argument("--verbose", "-v", action="store_true")
    remove_parser.add_argument("--json", action="store_true", help="Print result as JSON")

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
    edit_parser.add_argument(
        "--min-silence",
        type=float,
        default=1.5,
        metavar="SEC",
        help="Minimum silence duration to remove, in seconds (default 1.5)",
    )
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

    # --- session (undo/redo edit history) ---
    session_parser = subparsers.add_parser(
        "session",
        help="Undo/redo history over a chain of edits to one file",
    )
    session_sub = session_parser.add_subparsers(dest="session_command", help="Session commands")

    session_start_parser = session_sub.add_parser("start", help="Begin a new edit session")
    session_start_parser.add_argument("source", help="Source media file path")
    session_start_parser.add_argument(
        "--session-id",
        dest="session_id",
        help="Explicit session id (resets that session to just the source if it already exists)",
    )
    session_start_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_undo_parser = session_sub.add_parser("undo", help="Step back one edit")
    session_undo_parser.add_argument("session_id", help="Session id")
    session_undo_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_redo_parser = session_sub.add_parser("redo", help="Re-apply the most recently undone edit")
    session_redo_parser.add_argument("session_id", help="Session id")
    session_redo_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_reset_parser = session_sub.add_parser(
        "reset", help="Discard all edit history and jump back to the original source"
    )
    session_reset_parser.add_argument("session_id", help="Session id")
    session_reset_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_jump_parser = session_sub.add_parser(
        "jump", help="Jump directly to an arbitrary point in history (no stack loss)"
    )
    session_jump_parser.add_argument("session_id", help="Session id")
    session_jump_parser.add_argument(
        "index",
        type=int,
        help=(
            "Target position: -1 for the original source, 0..N-1 for a history "
            "entry (0-indexed, matching each entry's own 'index' from history)"
        ),
    )
    session_jump_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_history_parser = session_sub.add_parser("history", help="List all recorded edits")
    session_history_parser.add_argument("session_id", help="Session id")
    session_history_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_end_parser = session_sub.add_parser("end", help="Delete a session's on-disk journal")
    session_end_parser.add_argument("session_id", help="Session id")
    session_end_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    session_prune_parser = session_sub.add_parser(
        "prune", help="Delete abandoned sessions untouched for longer than --max-age-seconds"
    )
    session_prune_parser.add_argument(
        "--max-age-seconds",
        dest="max_age_seconds",
        type=float,
        default=None,
        metavar="SECS",
        help="Prune sessions untouched for longer than this (default: 7 days)",
    )
    session_prune_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # --- apply (YAML plan runner) ---
    apply_parser = subparsers.add_parser(
        "apply",
        help="Run a YAML-declared sequence of edits (source/steps/output)",
    )
    apply_parser.add_argument("plan", help="Path to a YAML plan file")
    apply_parser.add_argument(
        "--no-session",
        action="store_true",
        help="Do not create/record a session journal for this run (overrides session.record_history)",
    )
    apply_parser.add_argument("--json", action="store_true", help="Print result as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "probe":
            return cmd_probe(args)
        elif args.command == "convert":
            return cmd_convert(args)
        elif args.command == "demix":
            return cmd_demix(args)
        elif args.command == "concat":
            return cmd_concat(args)
        elif args.command == "conform":
            return cmd_conform(args)
        elif args.command == "normalize":
            return cmd_normalize(args)
        elif args.command == "master":
            return cmd_master(args)
        elif args.command == "denoise":
            return cmd_denoise(args)
        elif args.command == "word-gaps":
            return cmd_word_gaps(args)
        elif args.command == "transcribe":
            return cmd_transcribe(args)
        elif args.command == "extract-text":
            return cmd_extract_text(args)
        elif args.command == "plan":
            return cmd_plan(args)
        elif args.command == "edit":
            return cmd_edit(args)
        elif args.command == "remove":
            return cmd_remove(args)
        elif args.command == "trim":
            return cmd_trim(args)
        elif args.command == "eval":
            return cmd_eval(args)
        elif args.command == "session":
            return cmd_session(args)
        elif args.command == "apply":
            return cmd_apply(args)
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


def cmd_demix(args):
    import shutil as _shutil

    from ._demix import isolate_vocals

    p = Path(args.input)
    vocals_output = args.vocals_output or str(p.with_name(f"{p.stem}.vocals.wav"))
    instruments_output = args.instruments_output or str(p.with_name(f"{p.stem}.instruments.wav"))

    vocals_path, instruments_path = isolate_vocals(
        args.input,
        model_name=args.model,
        device=args.device,
        verbose=args.verbose,
    )

    Path(vocals_output).parent.mkdir(parents=True, exist_ok=True)
    Path(instruments_output).parent.mkdir(parents=True, exist_ok=True)
    _shutil.copyfile(vocals_path, vocals_output)
    _shutil.copyfile(instruments_path, instruments_output)

    if args.json:
        print(
            json.dumps(
                {"vocals_output": vocals_output, "instruments_output": instruments_output},
                indent=2,
            )
        )
    else:
        print(f"✓ Vocals → {vocals_output}")
        print(f"✓ Instruments → {instruments_output}")
    return 0


def cmd_concat(args):
    from .concat import concat_audio

    result = concat_audio(
        args.inputs,
        args.output,
        reencode=args.reencode,
        bitrate=args.bitrate,
        sample_rate=args.sample_rate,
        channels=args.channels,
        verbose=args.verbose,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "inputs": list(args.inputs),
                    "output_path": result,
                    "reencode": args.reencode,
                },
                indent=2,
            )
        )
    else:
        print(f"✓ Concatenated {len(args.inputs)} file(s) → {result}")
    return 0


def cmd_conform(args):
    from .conform import conform_audio

    result = conform_audio(
        args.input,
        args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        bitrate=args.bitrate,
        duration=args.duration,
        verbose=args.verbose,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "input": args.input,
                    "output_path": result,
                    "sample_rate": args.sample_rate,
                    "channels": args.channels,
                    "duration": args.duration,
                },
                indent=2,
            )
        )
    else:
        print(f"✓ Conformed → {result}")
    return 0


def cmd_normalize(args):
    from .normalize import optimize_audio_volume

    output = args.output
    if not output and not args.in_place:
        p = Path(args.input)
        output = str(p.with_name(f"{p.stem}.norm{p.suffix}"))

    result = optimize_audio_volume(
        args.input,
        output,
        in_place=args.in_place,
        force=args.force,
        mean_threshold=args.mean_threshold,
        max_threshold=args.max_threshold,
        target_lufs=args.target_lufs,
        true_peak_db=args.true_peak,
        lra=args.lra,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "path": result.path,
                    "mean_db": result.mean_db,
                    "max_db": result.max_db,
                    "normalized": result.normalized,
                    "target_lufs": result.target_lufs,
                    "true_peak_db": result.true_peak_db,
                },
                indent=2,
            )
        )
    else:
        print(f"mean_volume: {result.mean_db:.1f} dB  max_volume: {result.max_db:.1f} dB")
        if result.normalized:
            print(f"✓ Normalised → {result.path} (target {result.target_lufs} LUFS)")
        else:
            print(f"✓ Volume OK — no normalisation needed ({result.path})")
    return 0


def cmd_master(args):
    from .master import master_audio

    result = master_audio(
        args.input,
        args.output,
        preset=args.preset,
        target_lufs=args.target_lufs,
        true_peak_db=args.true_peak,
        lra=args.lra,
        sample_rate=args.sample_rate,
        channels=args.channels,
        bitrate=args.bitrate,
        chain=args.chain,
        verbose=args.verbose,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "path": result.path,
                    "preset": result.preset,
                    "chain": result.chain,
                    "target_lufs": result.target_lufs,
                    "true_peak_db": result.true_peak_db,
                    "normalized": result.normalized,
                    "stats": {
                        "input_i": result.stats.input_i,
                        "input_tp": result.stats.input_tp,
                        "input_lra": result.stats.input_lra,
                        "input_thresh": result.stats.input_thresh,
                        "target_offset": result.stats.target_offset,
                    },
                },
                indent=2,
            )
        )
    else:
        s = result.stats
        print(
            f"input: {s.input_i:.1f} LUFS  peak: {s.input_tp:.1f} dBTP  LRA: {s.input_lra:.1f} LU"
        )
        if result.normalized:
            print(
                f"✓ Mastered ({result.preset}) → {result.path} "
                f"(target {result.target_lufs:g} LUFS / {result.true_peak_db:g} dBTP)"
            )
        else:
            print(f"✓ Silent input — transcoded without loudness normalisation → {result.path}")
    return 0


def cmd_denoise(args):
    from .denoise import denoise_audio

    result = denoise_audio(
        args.input,
        args.output,
        noise_reduction=args.noise_reduction,
        noise_floor=args.noise_floor,
        track_noise=not args.no_track_noise,
        bitrate=args.bitrate,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"✓ Denoised → {result.output_path}")
    return 0


def cmd_word_gaps(args):
    from .models import TranscriptResult
    from .word_gaps import shorten_word_gaps

    transcript = None
    if args.transcript:
        transcript = TranscriptResult.from_dict(
            json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        )

    result = shorten_word_gaps(
        args.input,
        args.output,
        transcript=transcript,
        use_local=args.local,
        language=args.language,
        model=args.model,
        threshold=args.threshold,
        target=args.target,
        reencode=args.reencode,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"✓ Shortened {result.artifacts['gaps_shortened']} gap(s) "
            f"(> {args.threshold}s → {args.target}s) → {result.output_path}"
        )
    return 0


def cmd_extract_text(args):
    from .models import TranscriptResult

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Transcript not found: {args.input}")

    tr = TranscriptResult.from_dict(json.loads(inp.read_text(encoding="utf-8")))
    output = args.output or str(inp.with_suffix(".txt"))
    Path(output).write_text(tr.text, encoding="utf-8")
    print(f"Saved to: {output}")
    return 0


def cmd_transcribe(args):
    from .transcribe import transcribe_audio

    result = transcribe_audio(
        args.input,
        use_local=args.local,
        language=args.language,
        model=args.model,
        speed=args.speed,
        vad_filter=args.vad_filter,
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

    output_path = args.output
    if not output_path:
        p = Path(args.input)
        if output_format == "json":
            output_path = str(p.parent / f"{p.stem}.transcript.json")
        elif output_format == "srt":
            output_path = str(p.parent / f"{p.stem}.srt")
        elif output_format == "txt":
            output_path = str(p.parent / f"{p.stem}.txt")

    if output_path:
        with open(output_path, "w") as f:
            f.write(content)
        print(f"Saved to: {output_path}")
    else:
        print(content)
    return 0


def cmd_remove(args):
    from .models import TranscriptResult
    from .remove_ranges import remove_time_ranges

    ranges: list[str] = list(args.range or [])
    if args.cut_from is not None or args.cut_to is not None:
        if args.cut_from is None or args.cut_to is None:
            print("Error: --from and --to must be used together", file=sys.stderr)
            return 1
        ranges.append(f"{args.cut_from}-{args.cut_to}")

    if not ranges:
        print("Error: provide --range START-END and/or --from TIME --to TIME", file=sys.stderr)
        return 1

    transcript = None
    if args.transcript:
        transcript = TranscriptResult.from_dict(
            json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        )

    result = remove_time_ranges(
        args.input,
        ranges,
        output_path=args.output,
        reencode=args.reencode,
        verbose=args.verbose,
        transcript=transcript,
        refine_boundaries=not args.no_refine_boundaries,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        plan = result.plan
        print(f"✓ Removed {plan.removed_duration:.1f}s → {result.output_path}")
        print(
            f"  {plan.original_duration:.1f}s → {plan.edited_duration:.1f}s "
            f"({len(ranges)} range(s))"
        )
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
            min_silence=args.min_silence,
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


def cmd_session(args):
    if not getattr(args, "session_command", None):
        print(
            "Error: specify a session command (start, undo, redo, reset, jump, history, end, prune)",
            file=sys.stderr,
        )
        return 1

    if args.session_command == "start":
        return cmd_session_start(args)
    elif args.session_command == "undo":
        return cmd_session_undo(args)
    elif args.session_command == "redo":
        return cmd_session_redo(args)
    elif args.session_command == "reset":
        return cmd_session_reset(args)
    elif args.session_command == "jump":
        return cmd_session_jump(args)
    elif args.session_command == "history":
        return cmd_session_history(args)
    elif args.session_command == "end":
        return cmd_session_end(args)
    elif args.session_command == "prune":
        return cmd_session_prune(args)

    print(f"Error: unknown session command: {args.session_command}", file=sys.stderr)
    return 1


def cmd_session_start(args):
    from .session import current_path, start_session

    sid = start_session(args.source, session_id=args.session_id)

    if args.json:
        print(
            json.dumps(
                {
                    "session_id": sid,
                    "source_path": args.source,
                    "current_path": current_path(sid),
                },
                indent=2,
            )
        )
    else:
        print(f"✓ Session started: {sid}")
        print(f"  source: {args.source}")
    return 0


def cmd_session_undo(args):
    from .session import session_exists, undo

    if not session_exists(args.session_id):
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        return 1

    path = undo(args.session_id)
    if path is None:
        print("Nothing to undo.")
        return 0

    if args.json:
        print(json.dumps({"session_id": args.session_id, "path": path}, indent=2))
    else:
        print(f"✓ Reverted to: {path}")
    return 0


def cmd_session_redo(args):
    from .session import redo, session_exists

    if not session_exists(args.session_id):
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        return 1

    path = redo(args.session_id)
    if path is None:
        print("Nothing to redo.")
        return 0

    if args.json:
        print(json.dumps({"session_id": args.session_id, "path": path}, indent=2))
    else:
        print(f"✓ Re-applied: {path}")
    return 0


def cmd_session_reset(args):
    from .session import reset, session_exists

    if not session_exists(args.session_id):
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        return 1

    path = reset(args.session_id)
    if args.json:
        print(json.dumps({"session_id": args.session_id, "path": path}, indent=2))
    else:
        print(f"✓ Reset to original source: {path}")
    return 0


def cmd_session_jump(args):
    from .session import jump_to, session_exists

    if not session_exists(args.session_id):
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        return 1

    path = jump_to(args.session_id, args.index)
    if path is None:
        print("Nothing to jump to.")
        return 0

    if args.json:
        print(
            json.dumps(
                {"session_id": args.session_id, "index": args.index, "path": path},
                indent=2,
            )
        )
    else:
        print(f"✓ Jumped to: {path}")
    return 0


def cmd_session_history(args):
    from .session import current_path, history, session_exists

    if not session_exists(args.session_id):
        print(f"Unknown session: {args.session_id}", file=sys.stderr)
        return 1

    entries = history(args.session_id)
    cur = current_path(args.session_id)

    if args.json:
        print(
            json.dumps(
                {"session_id": args.session_id, "current_path": cur, "history": entries},
                indent=2,
            )
        )
        return 0

    if not entries:
        print("No edits recorded yet.")
    else:
        for entry in entries:
            marker = "*" if entry["active"] else " "
            print(
                f"[{marker}] {entry['index']}: {entry['operation']} -> {entry['path']} "
                f"({entry['timestamp']})"
            )
    print(f"Current: {cur}")
    return 0


def cmd_session_end(args):
    from .session import end_session

    removed = end_session(args.session_id)
    if args.json:
        print(json.dumps({"session_id": args.session_id, "removed": removed}, indent=2))
    elif removed:
        print(f"✓ Session ended: {args.session_id}")
    else:
        print(f"Session already gone: {args.session_id}")
    return 0


def cmd_session_prune(args):
    from .session import DEFAULT_SESSION_MAX_AGE_SECONDS, prune_sessions

    max_age = args.max_age_seconds
    removed = prune_sessions(max_age_seconds=max_age)
    effective_max_age = DEFAULT_SESSION_MAX_AGE_SECONDS if max_age is None else max_age

    if args.json:
        print(json.dumps({"removed": removed, "max_age_seconds": effective_max_age}, indent=2))
    else:
        print(f"✓ Pruned {removed} abandoned session(s) untouched for over {effective_max_age:g}s")
    return 0


def cmd_apply(args):
    from .yaml_plan import run_plan

    result = run_plan(args.plan, no_session=args.no_session)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for step in result["steps"]:
            detail = ", ".join(
                f"{k}={v}" for k, v in step.items() if k not in ("index", "op")
            )
            print(f"✓ [{step['index']}] {step['op']}" + (f" ({detail})" if detail else ""))
        if result.get("session_id"):
            print(f"Session: {result['session_id']}")
        print(f"✓ Applied {len(result['steps'])} step(s) → {result['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
