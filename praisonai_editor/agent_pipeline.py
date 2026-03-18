"""Prompt-based media editing via PraisonAI Agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import EditResult


def prompt_edit(
    input_path: str,
    prompt: str,
    output_path: Optional[str] = None,
    *,
    use_local_whisper: bool = False,
    verbose: bool = False,
) -> EditResult:
    """Edit a media file based on a natural language prompt.

    Uses a PraisonAI Agent to interpret the prompt, analyze the transcript,
    and create an edit plan.

    Args:
        input_path: Path to input media file
        prompt: Natural language editing instructions
            e.g. "Remove the intro and any off-topic discussion about weather"
        output_path: Path for output (auto-generated if None)
        use_local_whisper: Use local Whisper model
        verbose: Print progress

    Returns:
        EditResult with editing results

    Requires:
        pip install 'praisonai-editor[agent]'
    """
    try:
        from praisonaiagents import Agent
    except ImportError:
        raise ImportError(
            "praisonaiagents required for prompt-based editing. "
            "Install with: pip install 'praisonai-editor[agent]'"
        )

    from .probe import FFmpegProber
    from .transcribe import OpenAITranscriber, LocalTranscriber
    from .render import FFmpegAudioRenderer, FFmpegVideoRenderer
    from .pipeline import VIDEO_EXTENSIONS
    from .models import EditPlan, Segment

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_file.suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS

    if output_path is None:
        output_path = str(input_file.parent / f"{input_file.stem}_edited{input_file.suffix}")

    # Step 1: Probe
    if verbose:
        print(f"[1/4] Probing: {input_path}")
    prober = FFmpegProber()
    probe = prober.probe(input_path)

    # Step 2: Transcribe
    if verbose:
        print(f"[2/4] Transcribing ({probe.duration:.1f}s)...")
    if use_local_whisper:
        transcriber = LocalTranscriber()
    else:
        transcriber = OpenAITranscriber()
    transcript = transcriber.transcribe(input_path)

    # Step 3: Use Agent to create edit plan from prompt
    if verbose:
        print(f"[3/4] AI planning edits based on prompt: '{prompt}'")

    plan_agent = Agent(
        instructions=f"""You are a media editor AI. Given a transcript with word-level timestamps
and an editing instruction, produce a JSON edit plan.

The transcript is:
{transcript.text}

Word timestamps (first 100):
{json.dumps([w.to_dict() for w in transcript.words[:100]], indent=2)}

Total duration: {probe.duration:.2f} seconds

The user's editing instruction is:
"{prompt}"

Respond with ONLY a valid JSON object in this exact format:
{{
  "segments": [
    {{"start": 0.0, "end": 5.0, "action": "keep", "reason": "intro content", "category": "content"}},
    {{"start": 5.0, "end": 10.0, "action": "remove", "reason": "off-topic", "category": "tangent"}}
  ]
}}

Rules:
- Segments must cover the full duration from 0 to {probe.duration:.2f}
- Each segment has "action" of "keep" or "remove"
- Segments must not overlap and must be in chronological order
- Be conservative: only remove what the user asked to remove
""",
        verbose=verbose,
    )

    try:
        result = plan_agent.start(
            f"Create an edit plan for this media file based on: {prompt}"
        )

        # Parse the agent's response as JSON
        result_text = str(result)

        # Extract JSON from the response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        plan_data = json.loads(result_text.strip())

        segments = []
        for seg in plan_data.get("segments", []):
            segments.append(Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                action=seg.get("action", "keep"),
                reason=seg.get("reason", ""),
                category=seg.get("category", "content"),
            ))

        removed = sum(s.end - s.start for s in segments if s.action == "remove")
        plan = EditPlan(
            segments=segments,
            original_duration=probe.duration,
            edited_duration=probe.duration - removed,
            removed_duration=removed,
        )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback to heuristic if agent fails
        if verbose:
            print(f"    ⚠ Agent plan parsing failed ({e}), falling back to heuristic")
        from .plan import HeuristicEditor
        editor = HeuristicEditor()
        plan = editor.create_plan(transcript, probe.duration)

    # Step 4: Render
    if verbose:
        print(f"[4/4] Rendering: {output_path}")
        print(f"    Keeping {plan.edited_duration:.1f}s / {plan.original_duration:.1f}s")

    if is_video:
        renderer = FFmpegVideoRenderer()
    else:
        renderer = FFmpegAudioRenderer()

    renderer.render(input_path, output_path, plan, verbose=verbose)

    if verbose:
        print(f"\n✓ Done! Output: {output_path}")

    return EditResult(
        input_path=input_path,
        output_path=output_path,
        probe=probe,
        transcript=transcript,
        plan=plan,
        success=True,
        artifacts={"output": output_path},
    )
