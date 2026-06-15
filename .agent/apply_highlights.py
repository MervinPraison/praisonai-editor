#!/usr/bin/env python3
"""Apply praisonaippt YAML slide highlights to biblerevelation HTML drafts.

Usage:
  python apply_highlights.py --yaml deck.yaml --html article.html
  python apply_highlights.py --yaml a.yaml --heading "Title A" --yaml b.yaml --heading "Title B" --html post.html
  python apply_highlights.py --layout great-faith --yaml great_faith.yaml --html draft1.html --html draft2.html
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

_AGENT = Path(__file__).resolve().parent
SCRIPTURE_START = '<!-- wp:heading -->\n<h2 class="wp-block-heading">📖 Scripture from the Slides'
TAKEAWAY = '<!-- wp:heading -->\n<h2 class="wp-block-heading">🎯 The Takeaway</h2>'
SEP = (
    '<!-- wp:separator -->\n<hr class="wp-block-separator has-alpha-channel-opacity"/>'
    "\n<!-- /wp:separator -->"
)

HL = {
    "orange": (
        '<mark style="background-color:#FFF3E0;color:#C2410C;'
        'padding:0 2px;border-radius:2px;"><strong>{t}</strong></mark>'
    ),
    "green": (
        '<mark style="background-color:#DCFCE7;color:#15803D;'
        'padding:0 2px;border-radius:2px;"><strong>{t}</strong></mark>'
    ),
    "gold": (
        '<mark style="background-color:#FEF9C3;color:#854D0E;'
        'padding:0 2px;border-radius:2px;"><strong>{t}</strong></mark>'
    ),
}


def wrap(text: str, color_key: str = "orange") -> str:
    return HL[color_key].format(t=text)


def norm_ref(ref: str) -> str:
    return ref.replace("–", "-").strip() if ref else ""


def hl_color_key(h) -> str:
    if not isinstance(h, dict):
        return "orange"
    c = h.get("color")
    if not c:
        return "orange"
    c = str(c).lower().strip()
    if c == "green":
        return "green"
    if c in ("#ffd700", "ffd700", "yellow", "gold"):
        return "gold"
    return "orange"


def apply_highlights(text: str, highlights) -> str:
    if not highlights or not text:
        return text
    items = []
    seen_text: set[str] = set()
    for h in highlights:
        if isinstance(h, str):
            key = h.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            items.append({"text": h, "key": "orange"})
        elif isinstance(h, dict) and h.get("text"):
            key = h["text"].lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            items.append({"text": h["text"], "key": hl_color_key(h)})
    items.sort(key=lambda x: len(x["text"]), reverse=True)
    out = text
    for item in items:
        pattern = re.compile(re.escape(item["text"]), re.IGNORECASE)
        out = pattern.sub(lambda m, k=item["key"]: wrap(m.group(0), k), out, count=1)
    return out


def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t.strip())
    t = re.sub(r"\b(\d{1,3})\s+(?=[A-Za-z'\"(])", "", t)
    return t


def bullet_list(text: str) -> str:
    items = [ln.strip() for ln in text.split("\n") if ln.strip()]
    lis = "".join(f"<li>{item}</li>" for item in items)
    return f'<!-- wp:list -->\n<ul class="wp-block-list">{lis}</ul>\n<!-- /wp:list -->'


def add_verse(blocks: list, ref: str, text: str, highlights) -> None:
    highlighted = apply_highlights(text, highlights)
    blocks.append(f"<!-- wp:paragraph -->\n<p><strong>{ref}</strong></p>\n<!-- /wp:paragraph -->")
    blocks.append(
        f'<!-- wp:quote -->\n<blockquote class="wp-block-quote"><p>{highlighted}</p></blockquote>\n<!-- /wp:quote -->'
    )


def render_section(blocks: list, section_title: str, verses: list) -> None:
    title = section_title.strip() or "Foundation"
    blocks.append(
        f'<!-- wp:heading {{"level":3}} -->\n<h3 class="wp-block-heading">{title}</h3>\n<!-- /wp:heading -->'
    )
    for v in verses:
        ref = v.get("reference", "")
        text = v.get("text", "")
        highlights = v.get("highlights") or []
        if v.get("list_type") == "bullet":
            blocks.append(bullet_list(text))
        elif ref or text:
            add_verse(blocks, ref or "Slide note", clean_text(text), highlights)


def build_deck_section(data: dict, deck_heading: str) -> str:
    blocks: list[str] = []
    blocks.append(
        f'<!-- wp:heading -->\n<h2 class="wp-block-heading">{deck_heading}</h2>\n<!-- /wp:heading -->'
    )
    for sec in data.get("sections", []):
        render_section(blocks, sec.get("section", ""), sec.get("verses", []))
    return "\n\n".join(blocks)


def build_deck_appendix(decks: list[tuple[dict, str]]) -> str:
    parts = [
        SEP,
        '<!-- wp:heading -->\n<h2 class="wp-block-heading">📖 Scripture from the Slides (Every Verse)</h2>\n<!-- /wp:heading -->',
        "<!-- wp:paragraph -->",
        f"<p>Every verse from the sermon slides, with highlights — "
        f"{wrap('orange default', 'orange')}, "
        f"{wrap('green key themes', 'green')}, "
        f"{wrap('gold emphasis', 'gold')}.</p>",
        "<!-- /wp:paragraph -->",
    ]
    body: list[str] = []
    for i, (data, heading) in enumerate(decks):
        if i:
            body.append(SEP)
        body.append(build_deck_section(data, heading))
    return "\n\n".join(parts + body)


def build_great_faith_section(data: dict) -> str:
    """Custom section order for great_faith.yaml (legacy layout)."""
    verse_entries = []
    for sec in data.get("sections", []):
        for v in sec.get("verses", []):
            ref = v.get("reference", "")
            if not ref:
                continue
            verse_entries.append((ref, clean_text(v.get("text", "")), v.get("highlights") or []))

    mat15_parts, mat15_hl = [], []
    for ref, text, hl in verse_entries:
        if norm_ref(ref).startswith("Matthew 15:22"):
            mat15_parts.append(text)
            mat15_hl.extend(hl)

    seen: set = set()
    mat15_hl_unique = []
    for h in mat15_hl:
        key = h if isinstance(h, str) else h.get("text")
        if key and key not in seen:
            seen.add(key)
            mat15_hl_unique.append(h)
    mat15_full = " ".join(mat15_parts)

    blocks: list[str] = []
    blocks.append(
        '<!-- wp:heading {"level":3} -->\n<h3 class="wp-block-heading">Foundation</h3>\n<!-- /wp:heading -->'
    )
    for ref, text, hl in verse_entries:
        if ref.startswith(("Romans 5:17", "Romans 5:19")):
            add_verse(blocks, ref, text, hl)

    blocks.append(
        '<!-- wp:heading {"level":3} -->\n<h3 class="wp-block-heading">🪖 The Centurion</h3>\n<!-- /wp:heading -->'
    )
    for ref, text, hl in verse_entries:
        if ref.startswith("Matthew 8:5") or ref == "Matthew 8:13 (NKJV)":
            add_verse(blocks, ref, text, hl)

    blocks.append(
        '<!-- wp:heading {"level":3} -->\n<h3 class="wp-block-heading">🙏 The Canaanite Woman</h3>\n<!-- /wp:heading -->'
    )
    add_verse(blocks, "Matthew 15:22–28 (NKJV)", mat15_full, mat15_hl_unique)

    for title, list_html, ref_filter in (
        (
            "1️⃣ They Didn't Wait for God",
            "<!-- wp:list -->\n<ul class=\"wp-block-list\">"
            "<li>🩸 Woman with the Issue of Blood</li>"
            "<li>🪖 Centurion</li><li>🙏 Canaanite</li></ul>\n<!-- /wp:list -->",
            lambda r, t, h: (r.startswith("Mark 5:27") and "29" in r) or r in ("Matthew 8:7 (NKJV)", "Matthew 15:22 (NKJV)"),
        ),
        (
            "2️⃣ They Knew the Power of God",
            "<!-- wp:list -->\n<ul class=\"wp-block-list\">"
            "<li>💥 God's Power Is Greater Than My Sin</li>"
            "<li>🍞 Even Crumbs Are Enough</li>"
            "<li>⚡ One Word Is Enough</li>"
            "<li>👗 Even if I touch the hem of the garment</li></ul>\n<!-- /wp:list -->",
            lambda r, t, h: r in ("Romans 5:20 (NKJV)", "Matthew 8:8 (NKJV)"),
        ),
        (
            "3️⃣ They Knew the Heart of God",
            "<!-- wp:list -->\n<ul class=\"wp-block-list\">"
            "<li>❤️ They Knew His Heart Is Fatherly</li>"
            "<li>🤒 Fatherly Heart Wants to Heal</li>"
            "<li>⚡ Fatherly Heart Responds Immediately</li>"
            "<li>👑 Fatherly Heart Wants His Children to Take Authority</li>"
            "<li>🎁 Fatherly Heart Wants His Children to Claim Their Blessing</li>"
            "</ul>\n<!-- /wp:list -->",
            lambda r, t, h: r == "Matthew 7:11 (NKJV)",
        ),
        (
            "4️⃣ They Knew the Gospel",
            "<!-- wp:list -->\n<ul class=\"wp-block-list\">"
            "<li>✝️ They heard about Jesus</li>"
            "<li>📖 Righteousness of God Revealed</li>"
            "<li>⚖️ Law is not of faith — Galatians 3:12</li>"
            "<li>✍️ Know the Author</li></ul>\n<!-- /wp:list -->",
            lambda r, t, h: r
            in (
                "Mark 5:27 (NKJV)",
                "Luke 7:3 (NKJV)",
                "Mark 7:25 (NKJV)",
                "Romans 5:5 (NKJV)",
                "Romans 5:8 (NKJV)",
                "1 John 4:10 (NKJV)",
            ),
        ),
    ):
        blocks.append(
            f'<!-- wp:heading {{"level":3}} -->\n<h3 class="wp-block-heading">{title}</h3>\n<!-- /wp:heading -->'
        )
        blocks.append(list_html)
        for ref, text, hl in verse_entries:
            if ref_filter(ref, text, hl):
                add_verse(blocks, ref, text, hl)

    add_verse(
        blocks,
        "Galatians 3:12 (NKJV)",
        'Yet the law is not of faith, but "The man who does them shall live by them."',
        [],
    )

    body = "\n\n".join(blocks)
    return f"""{SEP}

<!-- wp:heading -->
<h2 class="wp-block-heading">📖 Scripture from the Slides (Every Verse)</h2>
<!-- /wp:heading -->

<!-- wp:paragraph -->
<p>Every scripture from the Great Faith slides, in full (NKJV), with slide highlights — {wrap("orange default", "orange")}, {wrap("green key themes", "green")}, {wrap("gold law emphasis", "gold")}.</p>
<!-- /wp:paragraph -->

{body}"""


def collect_inline_map(*datas: dict) -> dict[str, str]:
    inline: dict[str, str] = {}
    for data in datas:
        for sec in data.get("sections", []):
            for v in sec.get("verses", []):
                ref = v.get("reference", "")
                if not ref or v.get("list_type") == "bullet":
                    continue
                inline[norm_ref(ref)] = apply_highlights(
                    clean_text(v.get("text", "")), v.get("highlights") or []
                )
    return inline


def patch_inline_quotes(html: str, inline_map: dict[str, str], end_idx: int) -> str:
    pre = html[:end_idx]

    def replacer(match: re.Match) -> str:
        ref = match.group(1)
        body = inline_map.get(norm_ref(ref))
        if not body:
            return match.group(0)
        return f"{match.group(2)}{body}{match.group(4)}"

    pattern = re.compile(
        r'<p><strong>([^<]+)</strong></p>\s*<!-- /wp:paragraph -->\s*'
        r'(<!-- wp:quote -->\s*<blockquote class="wp-block-quote"><p>)'
        r'(.*?)'
        r'(</p></blockquote>)',
        re.DOTALL,
    )
    return pattern.sub(replacer, pre) + html[end_idx:]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def default_heading(path: Path, data: dict) -> str:
    if title := data.get("title"):
        return str(title)
    return path.stem.replace("_", " ").title()


def patch_html(html: str, scripture: str, inline_map: dict[str, str]) -> str:
    end = html.find(TAKEAWAY)
    if end == -1:
        raise ValueError("Takeaway marker not found in HTML")

    start = html.find(SCRIPTURE_START)
    if start != -1 and start < end:
        html = html[:start] + scripture + f"\n\n{SEP}\n\n" + html[end:]
        patch_end = start
    else:
        html = html[:end] + scripture + f"\n\n{SEP}\n\n" + html[end:]
        patch_end = end

    return patch_inline_quotes(html, inline_map, patch_end)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply PPT YAML highlights to biblerevelation HTML")
    p.add_argument("--yaml", "-y", action="append", required=True, metavar="FILE", help="PPT YAML deck (repeatable)")
    p.add_argument("--heading", action="append", default=[], metavar="TEXT", help="Section heading per --yaml (optional)")
    p.add_argument("--html", action="append", required=True, metavar="FILE", help="HTML draft to update (repeatable)")
    p.add_argument(
        "--layout",
        choices=("deck", "great-faith"),
        default="deck",
        help="deck: generic YAML sections; great-faith: legacy Great Faith ordering",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    yaml_paths = [Path(y).expanduser().resolve() for y in args.yaml]
    html_paths = [Path(h).expanduser().resolve() for h in args.html]

    datas = [load_yaml(p) for p in yaml_paths]
    if args.layout == "great-faith":
        if len(datas) != 1:
            print("great-faith layout accepts exactly one --yaml", file=sys.stderr)
            return 1
        scripture = build_great_faith_section(datas[0])
    else:
        headings = list(args.heading)
        while len(headings) < len(yaml_paths):
            headings.append(default_heading(yaml_paths[len(headings)], datas[len(headings)]))
        decks = list(zip(datas, headings[: len(yaml_paths)]))
        scripture = build_deck_appendix(decks)

    inline_map = collect_inline_map(*datas)

    for path in html_paths:
        if not path.exists():
            print(f"Missing: {path}", file=sys.stderr)
            return 1
        updated = patch_html(path.read_text(encoding="utf-8"), scripture, inline_map)
        path.write_text(updated, encoding="utf-8")
        print(f"Updated {path.name} — {len(inline_map)} inline verse refs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
