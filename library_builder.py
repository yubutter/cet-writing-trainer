"""Build the personal learning library by asking the AI to curate the parsed
material section-by-section.

Usage:
    # Full run
    python library_builder.py

    # Test on one section only
    python library_builder.py --only-section "1.促进、培养、激励"

    # Limit number of sections (for sanity / cost control)
    python library_builder.py --limit 3

Outputs:
    library/library.json   structured (used by the rest of the app)
    library/library.md     human-readable, grouped by category > section
    library/dropped.md     dropped items with reasons, for your review
    library/.cache/*.json  per-batch cache so re-runs skip already-done work
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ai_client import AIClient, AIError
from prompts import CURATE_SYSTEM, CURATE_USER_TEMPLATE

ROOT = Path(__file__).parent
SRC_JSON = ROOT / "source" / "parsed.json"
OUT_DIR = ROOT / "library"
CACHE_DIR = OUT_DIR / ".cache"

# Tunable: max items per single AI call. Keep small enough that the model can
# reason carefully about each one, large enough to keep call count down.
BATCH_SIZE = 40

# Used to invalidate the cache when prompt text changes.
PROMPT_VERSION = "curate-v1"


def _items_block(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, start=1):
        tag = "[phrase]" if it["is_phrase"] else "[word]  "
        if it.get("is_example"):
            tag = "[example]"
        lines.append(f"{i:3d}. {tag}  {it['english']}  ——  {it['chinese']}")
    return "\n".join(lines)


def _batch_hash(category: str, section: str, items: list[dict]) -> str:
    payload = json.dumps(
        {
            "v": PROMPT_VERSION,
            "category": category,
            "section": section,
            "items": [(it["english"], it["chinese"]) for it in items],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _chunks(seq: list[dict], n: int) -> Iterable[list[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def curate_batch(client: AIClient, category: str, section: str, items: list[dict]) -> dict:
    """Send one batch to the AI and return ``{"kept": [...], "dropped": [...]}``.

    Output dicts each have an ``entry`` field that points back to the matching
    original parsed entry (looked up by exact ``english`` match)."""
    user = CURATE_USER_TEMPLATE.format(
        category=category or "(unspecified)",
        section=section or "(unspecified)",
        n=len(items),
        items=_items_block(items),
    )
    data = client.chat_json(
        [
            {"role": "system", "content": CURATE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=12000,
    )
    if not isinstance(data, dict) or "kept" not in data or "dropped" not in data:
        raise AIError(f"Curation JSON missing keys: {str(data)[:400]}")

    by_eng = {it["english"]: it for it in items}
    kept = []
    dropped = []
    for k in data.get("kept", []):
        eng = (k.get("english") or "").strip()
        entry = by_eng.get(eng)
        if entry is None:
            # Try case-insensitive fallback
            entry = next(
                (it for it in items if it["english"].lower() == eng.lower()),
                None,
            )
        if entry is None:
            # Skip silently: model hallucinated an item we didn't send.
            continue
        kept.append({"entry": entry})
    for d in data.get("dropped", []):
        eng = (d.get("english") or "").strip()
        entry = by_eng.get(eng) or next(
            (it for it in items if it["english"].lower() == eng.lower()), None
        )
        if entry is None:
            continue
        dropped.append({"entry": entry, "reason": (d.get("reason") or "").strip()})

    # Coverage check: warn if model dropped items entirely.
    covered = {x["entry"]["english"] for x in kept} | {
        x["entry"]["english"] for x in dropped
    }
    missing = [it for it in items if it["english"] not in covered]
    if missing:
        # Default missing items to "kept" to avoid silent loss.
        for m in missing:
            kept.append({"entry": m})

    return {"kept": kept, "dropped": dropped}


def build(
    only_section: str | None = None,
    limit: int | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> dict:
    if not SRC_JSON.exists():
        raise SystemExit(
            f"Missing {SRC_JSON}. Run `python md_parser.py` first."
        )
    entries: list[dict] = json.loads(SRC_JSON.read_text(encoding="utf-8"))

    # Group by (category, section)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        groups[(e["category"], e["section"])].append(e)

    section_keys = list(groups.keys())
    if only_section:
        section_keys = [k for k in section_keys if k[1] == only_section]
        if not section_keys:
            raise SystemExit(f"No section matched: {only_section!r}")
    if limit:
        section_keys = section_keys[:limit]

    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    client = AIClient()

    library: dict[tuple[str, str], dict] = {}
    total_in = 0
    total_kept = 0
    total_dropped = 0
    t0 = time.time()

    for idx, key in enumerate(section_keys, start=1):
        cat, sec = key
        items = groups[key]
        total_in += len(items)
        sec_kept: list[dict] = []
        sec_dropped: list[dict] = []

        # Chunk if section is large
        chunks = list(_chunks(items, BATCH_SIZE))
        for ci, chunk in enumerate(chunks, start=1):
            h = _batch_hash(cat, sec, chunk)
            cache_file = CACHE_DIR / f"{h}.json"
            if cache_file.exists() and not refresh:
                batch = json.loads(cache_file.read_text(encoding="utf-8"))
                src = "cache"
            else:
                if verbose:
                    print(
                        f"[{idx}/{len(section_keys)}] {cat} > {sec}  "
                        f"chunk {ci}/{len(chunks)}  ({len(chunk)} items) → calling AI..."
                    )
                try:
                    batch = curate_batch(client, cat, sec, chunk)
                except AIError as e:
                    print(f"  ! AIError, skipping chunk: {e}", file=sys.stderr)
                    # Keep all items un-curated rather than losing them
                    batch = {
                        "kept": [{"entry": it, "scenario": ""} for it in chunk],
                        "dropped": [],
                    }
                cache_file.write_text(
                    json.dumps(batch, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                src = "api"

            sec_kept.extend(batch["kept"])
            sec_dropped.extend(batch["dropped"])
            if verbose:
                ratio = len(batch["kept"]) / max(1, len(batch["kept"]) + len(batch["dropped"]))
                print(
                    f"    kept {len(batch['kept'])} / dropped {len(batch['dropped'])} "
                    f"(ratio {ratio:.0%}) [{src}]"
                )

        library[key] = {"kept": sec_kept, "dropped": sec_dropped}
        total_kept += len(sec_kept)
        total_dropped += len(sec_dropped)

    elapsed = time.time() - t0

    # ----- write outputs -----
    library_json = []
    for (cat, sec), bucket in library.items():
        for k in bucket["kept"]:
            e = k["entry"]
            library_json.append(
                {
                    "category": cat,
                    "section": sec,
                    "english": e["english"],
                    "chinese": e["chinese"],
                    "is_phrase": e["is_phrase"],
                    "is_example": e.get("is_example", False),
                }
            )

    (OUT_DIR / "library.json").write_text(
        json.dumps(library_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_library_md(library)
    _write_dropped_md(library)

    print()
    print(f"=== Done in {elapsed:.1f}s ===")
    print(f"  sections processed: {len(section_keys)}")
    print(f"  entries in:    {total_in}")
    print(f"  entries kept:  {total_kept}")
    print(f"  entries dropped: {total_dropped}")
    if total_in:
        print(f"  kept ratio:    {total_kept / total_in:.1%}")
    print(f"  → {OUT_DIR / 'library.json'}")
    print(f"  → {OUT_DIR / 'library.md'}")
    print(f"  → {OUT_DIR / 'dropped.md'}")

    return {"kept": total_kept, "dropped": total_dropped, "in": total_in}


def _write_library_md(library: dict) -> None:
    lines = ["# 我的英语学习库（AI 精选）", ""]
    # Preserve category order from first occurrence
    cat_order: list[str] = []
    sec_by_cat: dict[str, list[str]] = defaultdict(list)
    for (cat, sec) in library.keys():
        if cat not in cat_order:
            cat_order.append(cat)
        if sec not in sec_by_cat[cat]:
            sec_by_cat[cat].append(sec)

    for cat in cat_order:
        lines.append(f"## {cat}")
        for sec in sec_by_cat[cat]:
            bucket = library[(cat, sec)]
            if not bucket["kept"]:
                continue
            heading = f"### {sec}" if sec else "### （未分小节）"
            lines.append("")
            lines.append(heading)
            lines.append("")
            for k in bucket["kept"]:
                e = k["entry"]
                tag = "📌" if e["is_phrase"] else "•"
                lines.append(f"- {tag} **{e['english']}** —— {e['chinese']}")
            lines.append("")
        lines.append("")
    (OUT_DIR / "library.md").write_text("\n".join(lines), encoding="utf-8")


def _write_dropped_md(library: dict) -> None:
    lines = ["# 被剔除的条目（AI 判断为生僻 / 不实用）", ""]
    for (cat, sec), bucket in library.items():
        if not bucket["dropped"]:
            continue
        lines.append(f"## {cat} > {sec or '（未分小节）'}")
        lines.append("")
        for d in bucket["dropped"]:
            e = d["entry"]
            reason = d.get("reason", "")
            lines.append(f"- ~~{e['english']}~~ —— {e['chinese']}  _(剔除理由：{reason})_")
        lines.append("")
    (OUT_DIR / "dropped.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-section", help="Only process sections with this exact title")
    ap.add_argument("--limit", type=int, help="Stop after N (category,section) groups")
    ap.add_argument("--refresh", action="store_true", help="Ignore cache and re-call AI")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        only_section=args.only_section,
        limit=args.limit,
        refresh=args.refresh,
        verbose=not args.quiet,
    )
