"""Full pipeline definition — audio through biblerevelation.org publish.

When the user says **full pipeline**, run every phase below in order.
Skip a phase only if the user explicitly opts out or the artefact already exists.

Phases 1–4: podcast-phone-audio (or sermon-transcribe for raw WAV).
Phases 5–11: biblerevelation-sermon-articles (transcript-only path when no YAML).

See also:
- `.cursor/skills/podcast-phone-audio/SKILL.md`
- `~/.cursor/skills/biblerevelation-sermon-articles/pipeline.md`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelinePhase:
    id: str
    name: str
    skill: str
    required: bool = True
    trigger: str = ""


FULL_PIPELINE: tuple[PipelinePhase, ...] = (
    PipelinePhase(
        "1",
        "Audio pipeline (crop optional → denoise → EQ → silence → normalise)",
        "podcast-phone-audio",
        trigger="process_phone_podcast.sh or noise-qa --apply",
    ),
    PipelinePhase(
        "2",
        "Chunked transcribe (5-min chunks for >10 min)",
        "podcast-phone-audio",
        trigger="transcribe_recording.sh {STEM}",
    ),
    PipelinePhase(
        "3",
        "Sanity check (duration, word count, volume, noise QA)",
        "podcast-phone-audio",
    ),
    PipelinePhase(
        "4",
        "Deliver FINAL m4a + transcript txt/json",
        "podcast-phone-audio",
    ),
    PipelinePhase(
        "5",
        "Transcript inventory (/tmp or .agent/{slug}-transcript-inventory.md)",
        "biblerevelation-sermon-articles",
    ),
    PipelinePhase(
        "6",
        "Article HTML draft (.agent/biblerevelation-{slug}.html)",
        "biblerevelation-sermon-articles",
    ),
    PipelinePhase(
        "7",
        "audit_transcript_coverage.py — must PASS before publish",
        "biblerevelation-sermon-articles",
    ),
    PipelinePhase(
        "8",
        "Visual brief + featured image (+ episode art for Spotify)",
        "biblerevelation-sermon-articles",
    ),
    PipelinePhase(
        "9",
        "Publish to biblerevelation.org (praisonaiwp create/update)",
        "biblerevelation-create-post",
    ),
    PipelinePhase(
        "10",
        "Live verification (HTTP 200, no critical error)",
        "biblerevelation-sermon-articles",
    ),
    PipelinePhase(
        "11",
        "Spotify notes (.agent/{slug}-spotify.md) + report post URL/ID",
        "podcast-phone-audio",
    ),
)

# Phases after Spotify upload — sermon import + auto-link (hourly cron also runs this).
POST_SPOTIFY: tuple[PipelinePhase, ...] = (
    PipelinePhase(
        "12",
        "Spotify upload (episode art + description with article URL)",
        "podcast-phone-audio",
        required=False,
    ),
    PipelinePhase(
        "13",
        "RSS sync: br_sermon_run_rss_sync() — imports sermon + auto-links article",
        "link-sermon-transcript",
        required=False,
        trigger="wp eval 'echo br_sermon_run_rss_sync();' — links via article URL or title match",
    ),
)


@dataclass
class PipelineValidation:
    stem: str
    repo: Path
    phases: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "repo": str(self.repo),
            "complete": self.complete,
            "phases": self.phases,
        }


def describe_full_pipeline(*, include_post_spotify: bool = False) -> str:
    """Human-readable checklist for agents and CLI."""
    lines = ["Full pipeline — run all phases when user says 'full pipeline':", ""]
    for p in FULL_PIPELINE:
        req = "required" if p.required else "optional"
        lines.append(f"  {p.id}. {p.name} [{p.skill}, {req}]")
        if p.trigger:
            lines.append(f"      → {p.trigger}")
    if include_post_spotify:
        lines.append("")
        lines.append("After Spotify upload:")
        for p in POST_SPOTIFY:
            lines.append(f"  {p.id}. {p.name} [{p.skill}]")
    return "\n".join(lines)


def validate_stem(stem: str, repo: str | Path | None = None) -> PipelineValidation:
    """Check which full-pipeline artefacts exist for a given stem."""
    root = Path(repo) if repo else Path.cwd()
    v = PipelineValidation(stem=stem, repo=root)

    checks: list[tuple[str, str, Path | None]] = [
        ("1", "FINAL audio", root / f"{stem}_FINAL.m4a"),
        ("2", "Transcript JSON", root / f"{stem}_FINAL.transcript.json"),
        ("2", "Transcript TXT", root / f"{stem}_FINAL.transcript.txt"),
        ("5", "Transcript inventory", None),  # slug-dependent — skip glob below
        ("6", "Article HTML", None),
        ("11", "Spotify notes", None),
    ]

    for phase_id, label, path in checks:
        if path is None:
            continue
        ok = path.exists() or (path.is_symlink() and path.resolve().exists())
        v.phases.append({"phase": phase_id, "check": label, "path": str(path), "ok": ok})

    # Any inventory / article / spotify in .agent matching stem-ish patterns
    agent = root / ".agent"
    if agent.is_dir():
        for pattern, phase_id, label in (
            (f"*{stem}*inventory*.md", "5", "Transcript inventory"),
            ("biblerevelation-*.html", "6", "Article HTML (any slug)"),
            (f"*{stem}*spotify*.md", "11", "Spotify notes"),
        ):
            matches = list(agent.glob(pattern)) if "*" in pattern else []
            if phase_id == "6":
                matches = [p for p in agent.glob("biblerevelation-*.html")]
            ok = bool(matches)
            v.phases.append(
                {
                    "phase": phase_id,
                    "check": label,
                    "path": str(matches[0]) if matches else str(agent / pattern),
                    "ok": ok,
                }
            )

    required_ok = all(
        p["ok"]
        for p in v.phases
        if p["check"] in {"FINAL audio", "Transcript TXT"}
    )
    v.complete = required_ok and any(p["ok"] and p["phase"] == "6" for p in v.phases)
    return v
