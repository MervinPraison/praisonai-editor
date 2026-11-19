# Custom Editor

Write your own logic for creating edit plans (what to keep and remove).

## Protocol

```python
from praisonai_editor.protocols import Editor
from praisonai_editor.models import EditPlan, TranscriptResult

class MyEditor:
    def create_plan(
        self,
        transcript: TranscriptResult,
        duration: float,
        *,
        remove_fillers: bool = True,
        remove_repetitions: bool = True,
        remove_silence: bool = True,
        min_silence: float = 1.5,
    ) -> EditPlan:
        ...
```

## Example: Remove only first N seconds

```python
from praisonai_editor.models import EditPlan, Segment

class TrimIntroEditor:
    def __init__(self, intro_secs: float = 30.0):
        self.intro_secs = intro_secs

    def create_plan(self, transcript, duration, **kwargs):
        return EditPlan(
            segments=[
                Segment(0, self.intro_secs, "remove", "intro", "content"),
                Segment(self.intro_secs, duration, "keep", "main content", "content"),
            ],
            original_duration=duration,
            edited_duration=duration - self.intro_secs,
            removed_duration=self.intro_secs,
        )
```
