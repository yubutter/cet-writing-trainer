"""Generate practice exercises from the personal learning library.

Two exercise types:
  * Translation drill: AI picks 5 library expressions per question and writes
    Chinese sentences that elicit those expressions in English translation.
  * Essay outline drill: AI writes one CET-6-style prompt; the student writes
    an outline; AI then critiques the outline against the library.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ai_client import AIClient, AIError, _log
from prompts import (
    EXERCISE_TRANSLATION_SYSTEM,
    EXERCISE_TRANSLATION_USER,
    EXERCISE_ESSAY_SYSTEM,
    EXERCISE_ESSAY_USER,
    OUTLINE_REVIEW_SYSTEM,
    OUTLINE_REVIEW_USER,
)
from reviewer import load_library, _format_library_subset, select_relevant


# ---------------------------------------------------------------------------
# Library sampling
# ---------------------------------------------------------------------------

def sample_library(
    library: list[dict],
    *,
    n: int = 30,
    category_filter: str | None = None,
    phrase_bias: float = 0.7,
    seed: int | None = None,
) -> list[dict]:
    """Sample ``n`` entries from the library, biased toward phrases.

    ``phrase_bias`` ∈ [0, 1]: fraction of the sample that should be phrases.
    """
    rng = random.Random(seed)
    pool = [e for e in library if not category_filter or e.get("category") == category_filter]
    if not pool:
        return []
    phrases = [e for e in pool if e.get("is_phrase")]
    words = [e for e in pool if not e.get("is_phrase")]
    n_phrase = min(len(phrases), int(round(n * phrase_bias)))
    n_word = min(len(words), n - n_phrase)
    return rng.sample(phrases, n_phrase) + rng.sample(words, n_word)


def list_categories(library: list[dict]) -> list[str]:
    seen: list[str] = []
    for e in library:
        c = e.get("category", "")
        if c and c not in seen:
            seen.add(c) if hasattr(seen, "add") else seen.append(c)  # type: ignore
    return seen


# ---------------------------------------------------------------------------
# Translation exercises
# ---------------------------------------------------------------------------

def gen_translation(
    n: int = 5,
    *,
    category_filter: str | None = None,
    library: list[dict] | None = None,
    client: AIClient | None = None,
    seed: int | None = None,
) -> dict:
    """Return ``{"items": [{chinese, target_expressions, reference_translation, key_points}, ...]}``."""
    _log(f"gen_translation: n={n} category_filter={category_filter!r}")
    library = library if library is not None else load_library()
    _log(f"gen_translation: library loaded, {len(library)} entries")
    if not library:
        raise RuntimeError("Library is empty. Run library_builder.py first.")

    # Sample ≈ 6× n expressions so the model has choice room.
    pool_size = max(40, n * 8)
    pool = sample_library(
        library,
        n=pool_size,
        category_filter=category_filter,
        phrase_bias=0.75,
        seed=seed,
    )
    user = EXERCISE_TRANSLATION_USER.format(
        n=n,
        library_subset=_format_library_subset(pool),
    )
    sys_msg = EXERCISE_TRANSLATION_SYSTEM.format(n=n)
    client = client or AIClient()
    data = client.chat_json(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
        max_tokens=max(3000, 500 * n),
        disable_thinking=True,
    )
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != n:
        raise AIError(f"模型未返回 {n} 道翻译题，请重新生成。")
    for item in items:
        if not isinstance(item, dict) or not _has_translation_fields(item):
            raise AIError("模型返回的翻译题内容不完整，请重新生成。")
    return data


def _has_translation_fields(item: dict) -> bool:
    for key in ("chinese", "reference_translation"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            return False
    for key in ("target_expressions", "key_points"):
        values = item.get(key)
        if not isinstance(values, list) or not values:
            return False
        if not all(isinstance(value, str) and value.strip() for value in values):
            return False
    return True


def review_translation_attempt(
    item: dict,
    student_translation: str,
    *,
    library: list[dict] | None = None,
    client: AIClient | None = None,
) -> dict:
    """Reuse the essay reviewer on a translation attempt against the reference."""
    # Lazy import to avoid circular costs at module load
    from reviewer import review

    chinese = item.get("chinese", "")
    ref = item.get("reference_translation", "")
    targets = ", ".join(item.get("target_expressions", []))
    prompt_block = (
        f"中文原句：{chinese}\n"
        f"建议覆盖的高级表达：{targets}\n"
        f"参考译文：{ref}"
    )
    return review(
        student_translation,
        task_type="translation",
        prompt_text=prompt_block,
        library=library,
        client=client,
    )


# ---------------------------------------------------------------------------
# Essay outline exercises
# ---------------------------------------------------------------------------

def gen_essay_topic(
    *,
    library: list[dict] | None = None,
    client: AIClient | None = None,
    topic_hint: str | None = None,
    seed: int | None = None,
) -> dict:
    _log(f"gen_essay_topic: topic_hint={topic_hint!r}")
    library = library if library is not None else load_library()
    _log(f"gen_essay_topic: library loaded, {len(library)} entries")
    if not library:
        raise RuntimeError("Library is empty. Run library_builder.py first.")

    cats = list_categories(library)
    chosen = topic_hint
    if not chosen:
        rng = random.Random(seed)
        # Prefer the thematic (non-"高级XXX合集") categories for essay topics
        thematic = [c for c in cats if "合集" not in c and "作文" not in c]
        chosen = rng.choice(thematic or cats)
    user = EXERCISE_ESSAY_USER.format(topic_hint=chosen)
    client = client or AIClient()
    data = client.chat_json(
        [
            {"role": "system", "content": EXERCISE_ESSAY_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
        max_tokens=1500,
        disable_thinking=True,
    )
    if not isinstance(data, dict) or "prompt" not in data:
        raise AIError(f"Essay topic JSON shape unexpected: {str(data)[:400]}")
    data["_topic_hint"] = chosen
    return data


def review_outline(
    topic: dict,
    student_outline: str,
    *,
    library: list[dict] | None = None,
    client: AIClient | None = None,
) -> dict:
    """Critique the student's outline against the AI-generated essay topic."""
    library = library if library is not None else load_library()
    subset = select_relevant(
        library,
        f"{student_outline}\n{topic.get('prompt','')}\n{topic.get('topic_zh','')}",
        max_n=50,
        topic_hint=topic.get("topic_zh", ""),
    )
    expected = topic.get("expected_structure", {})
    expected_str = json.dumps(expected, ensure_ascii=False, indent=2)
    user = OUTLINE_REVIEW_USER.format(
        essay_prompt=topic.get("prompt", ""),
        topic_zh=topic.get("topic_zh", ""),
        expected_structure=expected_str,
        outline=student_outline.strip(),
        library_subset=_format_library_subset(subset) or "(学习库为空)",
    )
    client = client or AIClient()
    data = client.chat_json(
        [
            {"role": "system", "content": OUTLINE_REVIEW_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=4000,
    )
    if not isinstance(data, dict):
        raise AIError(f"Outline review JSON shape unexpected: {str(data)[:400]}")
    return data


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_translation_questions(items: list[dict]) -> str:
    lines = ["# 翻译练习", ""]
    for i, it in enumerate(items, start=1):
        lines.append(f"## 第 {i} 题")
        lines.append("")
        lines.append(f"**中文**：{it.get('chinese','')}")
        lines.append("")
        tgt = it.get("target_expressions") or []
        if tgt:
            lines.append(f"**建议使用的高级表达**：{', '.join(tgt)}")
            lines.append("")
    return "\n".join(lines)


def render_translation_keys(items: list[dict]) -> str:
    lines = ["# 翻译练习参考答案", ""]
    for i, it in enumerate(items, start=1):
        lines.append(f"## 第 {i} 题")
        lines.append("")
        lines.append(f"**中文**：{it.get('chinese','')}")
        lines.append("")
        lines.append(f"**参考译文**：{it.get('reference_translation','')}")
        lines.append("")
        pts = it.get("key_points") or []
        if pts:
            lines.append("**得分点**：")
            for p in pts:
                lines.append(f"- {p}")
            lines.append("")
    return "\n".join(lines)


def render_essay_topic(topic: dict) -> str:
    lines = ["# 作文题", ""]
    lines.append(f"## {topic.get('title','')}")
    lines.append("")
    lines.append(topic.get("prompt", ""))
    lines.append("")
    lines.append(f"**主题**：{topic.get('topic_zh','')}")
    lines.append("")
    es = topic.get("expected_structure") or {}
    if es:
        lines.append("## 建议结构（出题人提示）")
        lines.append("")
        if es.get("intro"):
            lines.append(f"- 开头：{es['intro']}")
        for b in es.get("body", []):
            lines.append(f"- 论点：{b}")
        if es.get("conclusion"):
            lines.append(f"- 结尾：{es['conclusion']}")
        lines.append("")
    return "\n".join(lines)


def render_outline_review(review_data: dict) -> str:
    lines = ["# 大纲批改", ""]
    lines.append(f"- **是否扣题**：{'是' if review_data.get('matches_topic') else '否'}")
    lines.append(f"- **逻辑分**：{review_data.get('logic_score','—')}")
    lines.append(f"- **结构分**：{review_data.get('structure_score','—')}")
    lines.append("")
    if review_data.get("strengths"):
        lines.append("## 亮点")
        for s in review_data["strengths"]:
            lines.append(f"- {s}")
        lines.append("")
    if review_data.get("weaknesses"):
        lines.append("## 不足")
        for s in review_data["weaknesses"]:
            lines.append(f"- {s}")
        lines.append("")
    if review_data.get("missed_points"):
        lines.append("## 漏掉的关键论点")
        for s in review_data["missed_points"]:
            lines.append(f"- {s}")
        lines.append("")
    ro = review_data.get("rewritten_outline") or {}
    if ro:
        lines.append("## 推荐大纲")
        lines.append("")
        if ro.get("intro"):
            lines.append(f"- 开头：{ro['intro']}")
        for b in ro.get("body", []):
            lines.append(f"- 论点：{b}")
        if ro.get("conclusion"):
            lines.append(f"- 结尾：{ro['conclusion']}")
        lines.append("")
    recs = review_data.get("recommended_expressions") or []
    if recs:
        lines.append("## 建议在正文中使用的学习库表达")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")
    return "\n".join(lines)
