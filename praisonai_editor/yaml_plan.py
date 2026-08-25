"""YAML-declared sequence of edits, replayed through the real edit session.

Named ``yaml_plan.py`` (not ``plan.py`` -- that name is already taken by the
existing heuristic edit-plan module, :mod:`praisonai_editor.plan`).

A YAML file declares a ``source`` media file, an ordered ``steps`` list (each
``op: <name>`` + ``params: {...}`` mapping straight onto one of this
package's real functions), and an ``output`` path for the final result::

    source: talk.wav
    steps:
      - op: remove_ranges
        params:
          ranges: ["1:00-1:05"]
      - op: master
        params:
          preset: speech
          target_lufs: -14.0
    output: talk.mastered.m4a

Each step after the first implicitly consumes the PREVIOUS step's output as
its own source -- except ``concat``, which takes its own explicit
``params.sources: [...]`` list (multi-input by definition, it does not chain
from a single prior output). ``isolate_vocals`` produces two outputs
(vocals/instruments); the step must declare which one becomes the chained
"current" file via ``continue_with: vocals`` (or ``instruments``).

**Session integration is the point of this module.** Running a plan starts
(or resumes) a REAL :mod:`praisonai_editor.session` and records every step
into it via ``record_edit()`` as it happens -- exactly the way session.py's
own module docstring chains ``start_session`` / a real edit function /
``record_edit``. The session journal IS the replay log, so undo/redo/jump/
history all work "for free" afterward. Opt out with a top-level
``session.record_history: false`` in the YAML, or ``run_plan(..., no_session=True)``
(wired to the CLI's ``apply --no-session``) -- in that mode ``praisonai_editor.session``
is never imported and no journal file is created.

Usage:
    from praisonai_editor.yaml_plan import run_plan

    summary = run_plan("plan.yaml")
    print(summary["session_id"], summary["output_path"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

#: The 11 op names this module understands, mapped (in code) onto real
#: package functions -- see ``_run_step`` for the dispatch table.
OP_NAMES = (
    "transcribe",
    "preset_edit",
    "edit",
    "prompt_edit",
    "normalize",
    "master",
    "remove_ranges",
    "phrase_trim",
    "conform",
    "concat",
    "isolate_vocals",
    "convert",
    "denoise",
)


class PlanError(ValueError):
    """A malformed or invalid YAML plan (unknown op, missing field, bad shape).

    Subclasses ``ValueError`` so it is still caught cleanly by cli.py's
    existing top-level ``except Exception as e: print(f"Error: {e}")``
    convention -- this module does not invent a new error-reporting path.
    """


def load_plan(yaml_path_or_dict: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and validate a YAML plan (a file path, or an already-loaded dict).

    Raises:
        PlanError: For any structural problem (missing/invalid YAML, missing
            ``source``/``steps``, an unknown ``op``, or a step missing a
            field its op requires) -- never a raw ``KeyError`` or
            ``yaml.YAMLError``.
    """
    if isinstance(yaml_path_or_dict, dict):
        plan = copy.deepcopy(yaml_path_or_dict)
    else:
        path = Path(yaml_path_or_dict)
        if not path.is_file():
            raise PlanError(f"Plan file not found: {yaml_path_or_dict}")
        try:
            plan = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PlanError(f"Invalid YAML in {yaml_path_or_dict}: {exc}") from exc

    if not isinstance(plan, dict):
        raise PlanError("Plan must be a YAML mapping with 'source' and 'steps' keys")
    if not plan.get("source"):
        raise PlanError("Plan is missing required 'source' field")

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanError("Plan must have a non-empty 'steps' list")

    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not step.get("op"):
            raise PlanError(f"Step {i} is missing required 'op' field")
        op = step["op"]
        if op not in OP_NAMES:
            raise PlanError(
                f"Step {i}: unknown op {op!r} -- must be one of: {', '.join(OP_NAMES)}"
            )
        params = step.get("params") or {}
        if not isinstance(params, dict):
            raise PlanError(f"Step {i} ({op}): 'params' must be a mapping")
        if op == "concat" and not params.get("sources"):
            raise PlanError(f"Step {i} (concat): params.sources (a list) is required")
        if op == "isolate_vocals" and step.get("continue_with") not in ("vocals", "instruments"):
            raise PlanError(
                f"Step {i} (isolate_vocals): 'continue_with: vocals' or "
                "'continue_with: instruments' is required"
            )

    return plan


def _step_output(params: Dict[str, Any], plan_output: Optional[str], is_last: bool) -> Optional[str]:
    """Resolve a step's own output path: explicit params.output_path/output,
    else the plan's top-level 'output' if this is the last step, else None
    (let the underlying function pick its own default)."""
    if params.get("output_path"):
        return params["output_path"]
    if params.get("output"):
        return params["output"]
    if is_last and plan_output:
        return plan_output
    return None


def _run_transcribe(source, params, plan_output, is_last):
    from .transcribe import transcribe_audio
    import json as _json

    kwargs = {k: v for k, v in params.items() if k not in ("output_path", "output")}
    result = transcribe_audio(source, **kwargs)

    out_path = _step_output(params, plan_output, is_last)
    if out_path:
        ext = Path(out_path).suffix.lower()
        if ext == ".json":
            content = _json.dumps(result.to_dict(), indent=2)
        elif ext == ".txt":
            content = result.text
        else:
            content = result.to_srt()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(content, encoding="utf-8")

    # transcribe does not alter the media itself -- the chained "current"
    # stays whatever it was before this step.
    return {
        "current": source,
        "output_path": out_path,
        "record_params": dict(params),
        "result": {
            "output_path": out_path,
            "duration": result.duration,
            "language": result.language,
            "num_words": len(result.words),
        },
    }


def _run_edit_media(source, params, plan_output, is_last):
    from .pipeline import edit_media

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    result = edit_media(source, output_path=out_path, **kwargs)
    if not result.success:
        raise RuntimeError(f"edit_media step failed: {result.error}")
    return {
        "current": result.output_path,
        "output_path": result.output_path,
        "record_params": dict(params),
        "result": {"output_path": result.output_path},
    }


def _run_prompt_edit(source, params, plan_output, is_last):
    from .agent_pipeline import prompt_edit

    kwargs = dict(params)
    prompt = kwargs.pop("prompt", None)
    if not prompt:
        raise PlanError("op 'prompt_edit' requires params.prompt")
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    result = prompt_edit(source, prompt, output_path=out_path, **kwargs)
    if not result.success:
        raise RuntimeError(f"prompt_edit step failed: {result.error}")
    return {
        "current": result.output_path,
        "output_path": result.output_path,
        "record_params": dict(params),
        "result": {"output_path": result.output_path},
    }


def _run_normalize(source, params, plan_output, is_last):
    from .normalize import optimize_audio_volume

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)
    if out_path is None and not kwargs.get("in_place"):
        raise PlanError(
            "op 'normalize' requires params.output_path (or params.in_place: true), "
            "or must be the plan's final step with a top-level 'output'"
        )

    result = optimize_audio_volume(source, out_path, **kwargs)
    return {
        "current": result.path,
        "output_path": result.path,
        "record_params": dict(params),
        "result": {"path": result.path, "normalized": result.normalized},
    }


def _run_master(source, params, plan_output, is_last):
    from .master import master_audio

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    result = master_audio(source, out_path, **kwargs)
    return {
        "current": result.path,
        "output_path": result.path,
        "record_params": dict(params),
        "result": {"path": result.path, "preset": result.preset, "normalized": result.normalized},
    }


def _run_remove_ranges(source, params, plan_output, is_last):
    from .remove_ranges import remove_time_ranges

    kwargs = dict(params)
    ranges = kwargs.pop("ranges", None)
    if not ranges:
        raise PlanError("op 'remove_ranges' requires params.ranges (a list of START-END)")
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    # YAML's own list syntax (`[1.0, 1.5]`) parses to a plain Python list,
    # never a tuple -- but remove_time_ranges's parse_time_range() only
    # recognizes a *tuple* as "(start, end)" and otherwise falls through to
    # string-splitting, which mangles a list's str() representation into a
    # bogus time (e.g. "[1.0, 1.5]" split on "," gives "[1.0" as a
    # "timestamp"). A bare "11:53-12:43" string range is left untouched.
    # Studio's own job worker has the identical conversion for the same
    # reason (JSON arrays have the same tuple-vs-list gap as YAML lists).
    normalized_ranges = [
        tuple(r) if isinstance(r, (list, tuple)) else r
        for r in ranges
    ]

    result = remove_time_ranges(source, normalized_ranges, output_path=out_path, **kwargs)
    return {
        "current": result.output_path,
        "output_path": result.output_path,
        "record_params": dict(params),
        "result": {
            "output_path": result.output_path,
            "removed_duration": result.plan.removed_duration,
        },
    }


def _run_phrase_trim(source, params, plan_output, is_last):
    from .phrase_trim import trim_between_phrase_markers

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)
    if out_path is None:
        p = Path(source)
        out_path = str(p.parent / f"{p.stem}_trimmed{p.suffix}")

    path = trim_between_phrase_markers(source, out_path, **kwargs)
    return {
        "current": path,
        "output_path": path,
        "record_params": dict(params),
        "result": {"output_path": path},
    }


def _run_conform(source, params, plan_output, is_last):
    from .conform import conform_audio

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    path = conform_audio(source, out_path, **kwargs)
    return {
        "current": path,
        "output_path": path,
        "record_params": dict(params),
        "result": {"output_path": path},
    }


def _run_concat(source, params, plan_output, is_last):
    from .concat import concat_audio

    kwargs = dict(params)
    sources = kwargs.pop("sources", None)
    if not sources:
        raise PlanError("op 'concat' requires params.sources (a list of input files)")
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)
    if out_path is None:
        raise PlanError(
            "op 'concat' requires params.output_path, or must be the plan's "
            "final step with a top-level 'output'"
        )

    path = concat_audio(sources, out_path, **kwargs)
    return {
        "current": path,
        "output_path": path,
        "record_params": dict(params),
        "result": {"output_path": path, "sources": list(sources)},
    }


def _run_isolate_vocals(source, params, plan_output, is_last, continue_with):
    import shutil as _shutil

    from ._demix import isolate_vocals

    kwargs = dict(params)
    vocals_output = kwargs.pop("vocals_output", None)
    instruments_output = kwargs.pop("instruments_output", None)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)

    vocals_path, instruments_path = isolate_vocals(source, **kwargs)

    if vocals_output:
        Path(vocals_output).parent.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(vocals_path, vocals_output)
        vocals_path = vocals_output
    if instruments_output:
        Path(instruments_output).parent.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(instruments_path, instruments_output)
        instruments_path = instruments_output

    chosen = vocals_path if continue_with == "vocals" else instruments_path
    if is_last and plan_output:
        Path(plan_output).parent.mkdir(parents=True, exist_ok=True)
        _shutil.copyfile(chosen, plan_output)
        chosen = plan_output

    return {
        "current": chosen,
        "output_path": chosen,
        "record_params": {**dict(params), "continue_with": continue_with},
        "result": {
            "vocals_path": vocals_path,
            "instruments_path": instruments_path,
            "continue_with": continue_with,
        },
    }


def _run_convert(source, params, plan_output, is_last):
    from .convert import convert_media

    kwargs = dict(params)
    fmt = kwargs.pop("format", None)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)
    if out_path is None:
        p = Path(source)
        ext = fmt or p.suffix.lstrip(".") or "mp3"
        out_path = str(p.parent / f"{p.stem}.{ext}")

    path = convert_media(source, out_path, **kwargs)
    return {
        "current": path,
        "output_path": path,
        "record_params": dict(params),
        "result": {"output_path": path},
    }


def _run_denoise(source, params, plan_output, is_last):
    from .denoise import denoise_audio

    kwargs = dict(params)
    kwargs.pop("output_path", None)
    kwargs.pop("output", None)
    out_path = _step_output(params, plan_output, is_last)

    result = denoise_audio(source, out_path, **kwargs)
    return {
        "current": result.output_path,
        "output_path": result.output_path,
        "record_params": dict(params),
        "result": {"output_path": result.output_path, "artifacts": dict(result.artifacts or {})},
    }


def _run_step(op, current, params, plan_output, is_last, continue_with):
    if op == "isolate_vocals":
        return _run_isolate_vocals(current, params, plan_output, is_last, continue_with)
    if op == "concat":
        return _run_concat(current, params, plan_output, is_last)
    if op in ("preset_edit", "edit"):
        return _run_edit_media(current, params, plan_output, is_last)
    if op == "prompt_edit":
        return _run_prompt_edit(current, params, plan_output, is_last)
    if op == "transcribe":
        return _run_transcribe(current, params, plan_output, is_last)
    if op == "normalize":
        return _run_normalize(current, params, plan_output, is_last)
    if op == "master":
        return _run_master(current, params, plan_output, is_last)
    if op == "remove_ranges":
        return _run_remove_ranges(current, params, plan_output, is_last)
    if op == "phrase_trim":
        return _run_phrase_trim(current, params, plan_output, is_last)
    if op == "conform":
        return _run_conform(current, params, plan_output, is_last)
    if op == "convert":
        return _run_convert(current, params, plan_output, is_last)
    if op == "denoise":
        return _run_denoise(current, params, plan_output, is_last)
    raise PlanError(f"Unknown op: {op!r}")  # pragma: no cover -- load_plan already validates


def run_plan(yaml_path_or_dict: Union[str, Dict[str, Any]], *, no_session: bool = False) -> Dict[str, Any]:
    """Parse a YAML plan and execute its steps via this package's real functions.

    Session integration (the point of this module): unless opted out (via
    the plan's ``session.record_history: false`` or ``no_session=True``
    here), a real :mod:`praisonai_editor.session` is started (or resumed --
    see below) and every step is recorded into it with ``record_edit()`` as
    it happens, so ``session`` CLI undo/redo/jump/history all work
    afterward. When opted out, ``praisonai_editor.session`` is never
    imported and no journal file is created.

    Resuming: if the plan's ``session.id`` names an EXISTING session, its
    current on-disk state (``current_path``) is used as the starting point
    instead of the plan's ``source`` -- a real resume, continuing the
    session's own history. If that id does not exist yet (or none is
    given), a fresh session is started from ``source`` (session.py's own
    ``start_session`` semantics: an explicit id that does NOT yet exist
    behaves like a normal new session).

    Args:
        yaml_path_or_dict: Path to a YAML plan file, or an already-parsed dict.
        no_session: Force session-free execution regardless of what the
            plan's own ``session.record_history`` says (wired to the CLI's
            ``apply --no-session``).

    Returns:
        A summary dict: ``{"session_id": str | None, "source": str,
        "output_path": str, "steps": [...]}`` -- one entry in ``"steps"``
        per plan step, each carrying that op's own result fields.

    Raises:
        PlanError: The YAML is malformed, or references an unknown ``op``.
        RuntimeError: A step's underlying function failed or reported
            ``success=False``.
    """
    plan = load_plan(yaml_path_or_dict)
    steps = plan["steps"]
    source = plan["source"]
    plan_output = plan.get("output")
    session_cfg = plan.get("session") or {}
    record_history = bool(session_cfg.get("record_history", True)) and not no_session

    session_id = None
    current = source

    if record_history:
        from .session import session_exists, start_session, current_path

        explicit_id = session_cfg.get("id")
        if explicit_id and session_exists(explicit_id):
            session_id = explicit_id
            current = current_path(session_id) or source
        else:
            session_id = start_session(source, session_id=explicit_id)
            current = source

    step_summaries = []
    for i, step in enumerate(steps):
        op = step["op"]
        params = dict(step.get("params") or {})
        is_last = i == len(steps) - 1
        continue_with = step.get("continue_with")

        try:
            outcome = _run_step(op, current, params, plan_output, is_last, continue_with)
        except PlanError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Step {i} ({op}) failed: {exc}") from exc

        if record_history:
            from .session import record_edit

            record_path = outcome.get("output_path") or outcome["current"]
            record_edit(session_id, op, outcome["record_params"], record_path)

        current = outcome["current"]
        step_summaries.append({"index": i, "op": op, **outcome["result"]})

    return {
        "session_id": session_id,
        "source": source,
        "output_path": current,
        "steps": step_summaries,
    }
