# Custom Renderer

Replace FFmpeg rendering with your own engine (e.g., moviepy, pydub).

## Protocol

```python
from praisonai_editor.protocols import Renderer
from praisonai_editor.models import EditPlan

class MyRenderer:
    def render(
        self,
        input_path: str,
        output_path: str,
        plan: EditPlan,
        *,
        copy_codec: bool = True,
    ) -> str:
        ...
```

## Example: pydub renderer

```python
from pydub import AudioSegment
from praisonai_editor.models import EditPlan

class PydubRenderer:
    def render(self, input_path, output_path, plan, *, copy_codec=True):
        audio = AudioSegment.from_file(input_path)
        result = AudioSegment.empty()
        for seg in plan.get_keep_segments():
            result += audio[seg.start * 1000 : seg.end * 1000]
        result.export(output_path, format="mp3")
        return output_path
```
