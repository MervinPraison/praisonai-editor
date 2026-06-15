#!/usr/bin/env python3
"""Build biblerevelation HTML from legacy in-repo builders (100-fold, prevent-delay only).

New sermons: do NOT add recipe files. Write HTML directly from the transcript:

  .agent/biblerevelation-{slug}.html          # or {VIDEO_ID}_{start}_to_{end}.html

Then apply slide highlights:

  python apply_highlights.py --yaml deck.yaml --html .agent/your-draft.html

Usage (legacy builders in build_articles.py only):
  python build_post.py --slug 100-fold
  python build_post.py --slug prevent-delay
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_AGENT))

# Legacy only — new content should be .html drafts, not Python recipes
RECIPES: dict[str, tuple[str, str]] = {
    "100-fold": ("build_articles", "build_100_fold"),
    "prevent-delay": ("build_articles", "build_delay"),
}

OUTPUT_NAMES = {
    "100-fold": "biblerevelation-100-fold-blessings.html",
    "prevent-delay": "biblerevelation-how-to-prevent-delay.html",
}


def _load_builder(slug: str):
    mod_name, fn_name = RECIPES[slug]
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build biblerevelation HTML from legacy builders (prefer writing .html from transcript)"
    )
    p.add_argument("--slug", "-s", required=True, choices=sorted(RECIPES), help="Legacy recipe slug")
    p.add_argument("--output", "-o", metavar="FILE", help="Output HTML path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _AGENT / OUTPUT_NAMES[args.slug]
    )
    html = _load_builder(args.slug)()
    out.write_text(html, encoding="utf-8")
    print(
        f"Wrote {out} — h3: {html.count('level\":3')}, "
        f"tables: {html.count('wp:table')}, lists: {html.count('wp:list')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
