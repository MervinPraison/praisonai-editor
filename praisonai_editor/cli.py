"""CLI entry point for praisonai-editor.

Usage:
    praisonai-editor edit input.mp3 --output edited.mp3
    praisonai-editor transcribe input.mp3 --format srt
    praisonai-editor convert input.mp4 --format mp3
    praisonai-editor probe input.mp3
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
                             choices=["podcast", "meeting", "course", "clean"])
    edit_parser.add_argument("--prompt", help="Natural language editing instructions (uses AI agent)")
    edit_parser.add_argument("--no-fillers", action="store_true", help="Keep filler words")
    edit_parser.add_argument("--no-repetitions", action="store_true", help="Keep repetitions")
    edit_parser.add_argument("--no-silence", action="store_true", help="Keep silences")
    edit_parser.add_argument("--local", action="store_true", help="Use local whisper")
    edit_parser.add_argument("--reencode", action="store_true", help="Re-encode instead of copy")
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
            remove_fillers=not args.no_fillers,
            remove_repetitions=not args.no_repetitions,
            remove_silence=not args.no_silence,
            use_local_whisper=args.local,
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
