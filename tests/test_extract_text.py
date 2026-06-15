"""CLI extract-text from transcript JSON."""

import json
from pathlib import Path

from praisonai_editor.models import TranscriptResult


def test_extract_text_writes_plain_text(tmp_path):
    src = tmp_path / "clip.transcript.json"
    src.write_text(
        json.dumps(
            TranscriptResult(text="Hello world.", words=[], language="en", duration=1.0).to_dict()
        ),
        encoding="utf-8",
    )
    out = tmp_path / "clip.txt"

    from praisonai_editor.cli import cmd_extract_text

    class Args:
        input = str(src)
        output = str(out)

    assert cmd_extract_text(Args()) == 0
    assert out.read_text(encoding="utf-8") == "Hello world."
