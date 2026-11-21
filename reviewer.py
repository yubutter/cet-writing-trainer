"""Correct an English essay / translation and suggest upgrades from the
personal learning library.

The flow is two-stage *inside the same model call*:
  1) error-level corrections (vocab, grammar, syntax),
  2) library-aware upgrade suggestions.

To keep the prompt focused, we don't dump the whole library — we pick a
topic-relevant subset (≈ 60 entries) based on lightweight keyword overlap
between the student's text and the library entries (Chinese + English).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from ai_client import AIClient, AIError
from prompts import REVIEW_SYSTEM, REVIEW_USER_TEMPLATE

ROOT = Path(__file__).parent
LIB_PATH = ROOT / "library" / "library.json"
PARSED_PATH = ROOT / "source" / "parsed.json"

# ---------------------------------------------------------------------------
# Library loading + relevance selection
# ---------------------------------------------------------------------------

# Light tokenisation: split on non-letter and lower-case for English,
# and use char-level matching for Chinese.
EN_TOKEN_RE = re.compile(r"[A-Za-z]+")
CN_CHAR_RE = re.compile(r"[一-鿿]")

# Stopwords we don't want to drive relevance.
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "to", "for",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "us", "you", "your", "he", "she", "his", "her", "i", "me",
    "my", "do", "does", "did", "have", "has", "had", "not", "no", "so",
    "than", "then", "too", "very", "can", "could", "will", "would", "should",
    "may", "might", "must", "also", "such", "from", "into", "out", "up",
    "down", "about", "over", "under", "more", "most", "much", "many", "some",
    "any", "all", "each", "every", "other", "another", "one", "two", "first",
}


def load_library() -> list[dict]:
    if not LIB_PATH.exists():
        # Fall back to parsed (uncurated) so the app still works pre-build.
        if PARSED_PATH.exists():
            return json.loads(PARSED_PATH.read_text(encoding="utf-8"))
        return []
    return json.loads(LIB_PATH.read_text(encoding="utf-8"))


def _tokens(text: str) -> tuple[set[str], set[str]]:
    """Return (english_tokens, chinese_chars) for relevance scoring."""
    eng = {
        t.lower()
        for t in EN_TOKEN_RE.findall(text)
        if len(t) >= 3 and t.lower() not in EN_STOPWORDS
    }
    cn = set(CN_CHAR_RE.findall(text))
    return eng, cn


def select_relevant(
    library: list[dict],
    text: str,
    *,
    max_n: int = 60,
    topic_hint: str = "",
) -> list[dict]:
    """Pick the `max_n` library entries most relevant to ``text``.

    Scoring: token overlap with each entry's (english + chinese + section + category).
    Always includes a category-balanced floor so reviewers see breadth.
    """
    if not library:
        return []
    haystack = f"{text}\n{topic_hint}"
    q_eng, q_cn = _tokens(haystack)

    scored: list[tuple[float, dict]] = []
    for e in library:
        e_eng, e_cn = _tokens(
            f"{e.get('english','')} {e.get('chinese','')} "
            f"{e.get('section','')} {e.get('category','')}"
        )
        overlap_eng = len(q_eng & e_eng)
        overlap_cn = len(q_cn & e_cn)
        score = overlap_eng * 2.0 + overlap_cn * 0.4
        # small bonus for phrases over single words (more idiomatic suggestions)
        if e.get("is_phrase"):
            score += 0.2
        scored.append((score, e))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    top = [e for score, e in scored if score > 0][:max_n]
    # If we still have headroom, fill from zero-score pool to give breadth.
    if len(top) < max_n:
        for score, e in scored:
            if score == 0 and e not in top:
                top.append(e)
                if len(top) >= max_n:
                    break
    return top


def _format_library_subset(items: Iterable[dict]) -> str:
    by_cat: dict[str, list[dict]] = {}
    for e in items:
        by_cat.setdefault(e.get("category", ""), []).append(e)
    chunks = []
    for cat, group in by_cat.items():
        chunks.append(f"### {cat}")
        for e in group:
            tag = "[phrase]" if e.get("is_phrase") else "[word]  "
            chunks.append(
                f"- {tag} {e.get('english','')} —— {e.get('chinese','')}"
            )
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Public review entrypoint
# ---------------------------------------------------------------------------

def review(
    student_text: str,
    *,
    task_type: str = "essay",  # "essay" or "translation"
    prompt_text: str = "",  # the essay question or the Chinese source for translation
    library: list[dict] | None = None,
    client: AIClient | None = None,
) -> dict:
    """Run the full two-stage review and return the parsed JSON dict.

    Return shape: {corrections: [...], upgrades: [...], overall: {...}}
    """
    student_text = (student_text or "").strip()
    if not student_text:
        raise ValueError("Empty student text.")

    library = library if library is not None else load_library()
    subset = select_relevant(
        library, student_text, max_n=60, topic_hint=prompt_text
    )
    subset_block = _format_library_subset(subset) or "(学习库为空)"

    if task_type == "translation":
        prompt_section = (
            f"【中文原文（翻译题源文，需对照核查要点是否译全）】\n"
            f"{prompt_text or '(未提供中文原文)'}\n\n"
        )
    elif prompt_text:
        prompt_section = f"【作文题目】\n{prompt_text}\n\n"
    else:
        prompt_section = ""

    user_msg = REVIEW_USER_TEMPLATE.format(
        task_type=("作文" if task_type == "essay" else "中译英翻译"),
        prompt_section=prompt_section,
        student_text=student_text,
        library_subset=subset_block,
    )

    client = client or AIClient()
    data = client.chat_json(
        [
            {"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=12000,
    )
    if not isinstance(data, dict):
        raise AIError(f"Review JSON shape unexpected: {str(data)[:400]}")
    # Defensive defaults
    data.setdefault("corrections", [])
    data.setdefault("upgrades", [])
    data.setdefault("overall", {})
    return data


# ---------------------------------------------------------------------------
# Markdown rendering for GUI / file output
# ---------------------------------------------------------------------------

def render_markdown(
    result: dict,
    *,
    task_type: str = "essay",
    prompt_text: str = "",
    student_text: str = "",
) -> str:
    lines: list[str] = []
    lines.append(f"# 批改结果（{'作文' if task_type == 'essay' else '翻译'}）")
    lines.append("")
    if prompt_text:
        lines.append("## 题目 / 原文")
        lines.append("")
        lines.append("```")
        lines.append(prompt_text.strip())
        lines.append("```")
        lines.append("")
    if student_text:
        lines.append("## 你的提交")
        lines.append("")
        lines.append("```")
        lines.append(student_text.strip())
        lines.append("```")
        lines.append("")

    overall = result.get("overall") or {}
    lines.append("## 总评")
    lines.append("")
    band = overall.get("band", "—")
    lines.append(f"- **估算评分档（15 分制）**：{band}")
    if overall.get("highlight"):
        lines.append(f"- **亮点**：{overall['highlight']}")
        if overall.get("to_improve"):
            lines.append(f"- **改进**：{overall['to_improve']}")
    lines.append("")

    corrections = result.get("corrections") or []
    lines.append(f"## 错误纠正（{len(corrections)} 条）")
    lines.append("")
    if not corrections:
        lines.append("_未发现明显错误。_")
    else:
        lines.append("| # | 错误类型 | 原文 | 修正 |")
        lines.append("|---:|---|---|---|")
        for i, c in enumerate(corrections, start=1):
            orig = (c.get("original", "") or "").replace("|", "\\|").replace("\n", " ")
            fixed = (c.get("fixed", "") or "").replace("|", "\\|").replace("\n", " ")
            typ = (c.get("type", "") or "").replace("|", "\\|")
            lines.append(f"| {i} | {typ} | {orig} | {fixed} |")
    lines.append("")

    upgrades = result.get("upgrades") or []
    lines.append(f"## 学习库升级建议（{len(upgrades)} 条）")
    lines.append("")
    if not upgrades:
        lines.append("_学习库中没有更合适的升级表达，或原文表达已经足够地道。_")
    else:
        for i, u in enumerate(upgrades, start=1):
            orig = u.get("original", "")
            sug = u.get("suggested", "")
            reason = u.get("reason", "")
            lines.append(f"**{i}.** `{orig}` → **{sug}**  _（{reason}）_")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI for ad-hoc testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to a .txt/.md file containing the essay")
    ap.add_argument(
        "--type", choices=["essay", "translation"], default="essay"
    )
    ap.add_argument(
        "--prompt", default="", help="Essay question, or Chinese source for translation"
    )
    args = ap.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    result = review(text, task_type=args.type, prompt_text=args.prompt)
    md = render_markdown(
        result,
        task_type=args.type,
        prompt_text=args.prompt,
        student_text=text,
    )
    print(md)
