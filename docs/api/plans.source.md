# Edit Plans API

An `EditPlan` is a list of segments with `keep` or `remove` actions.

## `create_edit_plan()`

```python
from praisonai_editor.plan import create_edit_plan

plan = create_edit_plan(
    transcript,
    duration=1823.4,
    preset="podcast",        # or specify kwargs:
    remove_fillers=True,
    remove_repetitions=True,
    remove_silence=True,
    min_silence=1.5,
)
```

## `EditPlan`

```python
plan.segments            # list[Segment]
plan.original_duration   # float
plan.edited_duration     # float
plan.removed_duration    # float
plan.removal_summary     # {"filler": 12.3, "silence": 45.6, ...}

plan.get_keep_segments()   # list[Segment] with action="keep"
plan.get_remove_segments() # list[Segment] with action="remove"
```

## `Segment`

```python
seg.start       # float (seconds)
seg.end         # float (seconds)
seg.action      # "keep" | "remove"
seg.reason      # "Filler word: 'um'"
seg.category    # "filler" | "repetition" | "silence" | "content"
seg.confidence  # 0.0–1.0
```

## Individual detectors

```python
from praisonai_editor.plan import detect_fillers, detect_repetitions, detect_silences

fillers    = detect_fillers(transcript.words)
reps       = detect_repetitions(transcript.words, window=3)
silences   = detect_silences(transcript.words, duration, min_silence=1.5)
```
