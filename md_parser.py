"""Parse the markdown vocabulary book into structured JSON.

Output: source/parsed.json
Each entry: {category, section, english, chinese, is_phrase, is_example, line}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent
SRC_MD = ROOT / "writing_materials.md"
OUT_DIR = ROOT / "source"
OUT_JSON = OUT_DIR / "parsed.json"

# Headings that should NEVER be treated as a (category, section) container —
# they are TOC entries (end with dots + page number) or are decorative.
TOC_PAGE_RE = re.compile(r"\.\s*\.?\s*\d+\s*$")  # "... 52" / ".. . 52" / ". 88"
IMAGE_RE = re.compile(r"^!\[.*?\]\(.*?\)\s*$")
HEADING_RE = re.compile(r"^#+\s+(.+?)\s*$")
CHINESE_CHAR_RE = re.compile(r"[一-鿿]")
ENGLISH_CHAR_RE = re.compile(r"[A-Za-z]")
STAR_SUFFIX_RE = re.compile(r"[★☆]+\s*$")
LEADING_NUMBER_RE = re.compile(r"^\s*\d+[\.、]\s*")  # "1. " / "1、"

# A heading is recognized as a TOP-LEVEL category if its text matches one of:
#   - ALL-CAPS English token at start (PERSONALITY, EDUCATION, ...)
#   - Known Chinese top-level names from the TOC
TOP_LEVEL_CHINESE = {
    "作文写作指南",
    "高级动词合集",
    "高级形容词合集",
    "高级连词合集",
    "图表类作文",
}
TOP_LEVEL_ENG_PREFIX_RE = re.compile(r"^[A-Z]{3,}\b")

# Notes that appear between entries but are NOT entries themselves.
NOTE_LINE_RE = re.compile(
    r"^\s*("
    r"词块[:：]?"          # "词块：" / "词块"
    r"|适用于"              # "适用于六级 ..."
    r"|注[:：]"             # "注："
    r"|—+|-+|=+"            # decorative separators
    r")"
)


def _strip_decoration(s: str) -> str:
    s = STAR_SUFFIX_RE.sub("", s).strip()
    s = LEADING_NUMBER_RE.sub("", s)
    return s.strip()


def _looks_like_toc(text: str) -> bool:
    return bool(TOC_PAGE_RE.search(text))


def _is_top_level(heading_text: str) -> bool:
    t = heading_text.strip()
    if TOP_LEVEL_ENG_PREFIX_RE.match(t):
        return True
    bare = _strip_decoration(t)
    return bare in TOP_LEVEL_CHINESE


def _split_eng_cn(text: str) -> tuple[str, str] | None:
    """Split a line containing English + Chinese into (english, chinese).
    Strategy: find the *first* Chinese character; everything before is English,
    everything from it onward is Chinese (after stripping)."""
    m = CHINESE_CHAR_RE.search(text)
    if not m:
        return None
    eng = text[: m.start()].strip()
    cn = text[m.start() :].strip()
    # Clean up common artifacts
    eng = re.sub(r"\s+", " ", eng).strip(" .,:;-—")
    cn = re.sub(r"\s+", " ", cn).strip()
    if not eng or not cn:
        return None
    return eng, cn


def _is_phrase(english: str) -> bool:
    # Phrase if more than one space-separated token, treating "xx" placeholder as token.
    tokens = [t for t in english.split() if t]
    return len(tokens) > 1


def _is_example(english: str) -> bool:
    return english.lower().startswith(("e.g.", "eg.", "eg "))


def parse(md_path: Path = SRC_MD) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    raw_lines = text.splitlines()

    # Group into blocks separated by blank lines, remembering the original line no.
    blocks: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start_lineno = 0
    for idx, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            if current:
                blocks.append((start_lineno, current))
                current = []
            continue
        if not current:
            start_lineno = idx
        current.append(line.rstrip())
    if current:
        blocks.append((start_lineno, current))

    entries: list[dict] = []
    category = ""
    section = ""
    pending_english: str | None = None  # for cross-block wrapping
    pending_lineno: int = 0

    started_content = False  # flip to True once we pass the TOC

    for lineno, block in blocks:
        # Heading?
        if len(block) == 1 and (m := HEADING_RE.match(block[0])):
            head = m.group(1).strip()
            if _looks_like_toc(head):
                continue  # TOC heading
            if not started_content:
                started_content = True
            head_clean = _strip_decoration(head)
            if _is_top_level(head):
                category = head_clean
                section = ""  # reset section under new category
            else:
                section = head_clean
            pending_english = None
            continue

        if not started_content:
            # Inside TOC region — skip anything that doesn't look like content.
            # Most TOC lines have a trailing ".. <page>"; treat as non-content.
            if all(_looks_like_toc(l) or HEADING_RE.match(l) for l in block):
                continue
            # If we hit a block that *doesn't* look like TOC, treat it as content.
            started_content = True

        # Skip image-only blocks.
        if all(IMAGE_RE.match(l) for l in block):
            continue

        # Skip note-only blocks ("词块：" alone, "适用于六级..." alone, etc.).
        joined = " ".join(block).strip()
        if all(NOTE_LINE_RE.match(l) for l in block):
            continue
        # Block starting with note keyword + rest — strip the keyword prefix.
        joined = re.sub(r"^\s*词块[:：]\s*", "", joined)
        joined = re.sub(r"^\s*注[:：]\s*", "", joined)

        # Skip "适用于六级 ..." description blocks (they describe which exam used the topic).
        if joined.startswith("适用于"):
            continue

        # If joined has both English and Chinese → straight entry.
        has_cn = bool(CHINESE_CHAR_RE.search(joined))
        has_en = bool(ENGLISH_CHAR_RE.search(joined))

        if has_en and has_cn:
            # Resolve any pending English fragment first by attaching its CN.
            # If the joined block itself has both, treat any pending English as orphaned -> drop.
            split = _split_eng_cn(joined)
            if split:
                eng, cn = split
                # Guard: if `eng` is suspiciously empty after stripping leading
                # numbering like "1." we already removed that.
                entries.append(
                    {
                        "category": category,
                        "section": section,
                        "english": eng,
                        "chinese": cn,
                        "is_phrase": _is_phrase(eng),
                        "is_example": _is_example(eng),
                        "line": lineno,
                    }
                )
            pending_english = None
            continue

        if has_en and not has_cn:
            # Pure English block — could be a wrapped entry whose CN is in next block.
            pending_english = joined
            pending_lineno = lineno
            continue

        if has_cn and not has_en:
            # Pure Chinese block — close out pending English if any.
            if pending_english:
                eng = re.sub(r"\s+", " ", pending_english).strip(" .,:;-—")
                cn = re.sub(r"\s+", " ", joined).strip()
                if eng and cn:
                    entries.append(
                        {
                            "category": category,
                            "section": section,
                            "english": eng,
                            "chinese": cn,
                            "is_phrase": _is_phrase(eng),
                            "is_example": _is_example(eng),
                            "line": pending_lineno,
                        }
                    )
                pending_english = None
            # else: stray Chinese, ignore
            continue

        # Neither English nor Chinese (e.g. just punctuation) — ignore.
        continue

    return entries


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    entries = parse()
    OUT_JSON.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Stats
    by_cat: dict[str, int] = {}
    by_section: dict[tuple[str, str], int] = {}
    phrase_n = sum(1 for e in entries if e["is_phrase"])
    example_n = sum(1 for e in entries if e["is_example"])
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
        key = (e["category"], e["section"])
        by_section[key] = by_section.get(key, 0) + 1

    print(f"Parsed {len(entries)} entries → {OUT_JSON.relative_to(ROOT)}")
    print(f"  single-word: {len(entries) - phrase_n}")
    print(f"  phrases:     {phrase_n}")
    print(f"  examples:    {example_n}")
    print(f"  categories:  {len(by_cat)}")
    print(f"  sections:    {len(by_section)}")
    print()
    print("Top categories by entry count:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:5d}  {cat}")
    print()
    print("Sample sections (showing 3 entries each, first 5 sections):")
    seen = 0
    for (cat, sec), _n in sorted(by_section.items()):
        if seen >= 5:
            break
        if not sec:
            continue
        sample = [e for e in entries if e["category"] == cat and e["section"] == sec][:3]
        print(f"  [{cat} > {sec}]")
        for e in sample:
            tag = "P" if e["is_phrase"] else "W"
            ex = "*" if e["is_example"] else " "
            print(f"    {tag}{ex} {e['english']}  —  {e['chinese']}")
        print()
        seen += 1


if __name__ == "__main__":
    main()
