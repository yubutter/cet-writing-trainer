"""PySide6 GUI for the personal English learning + correction toolkit.

Tabs:
  1. 学习库     – browse / search / filter the curated library
  2. 批改       – submit an essay or translation, get a corrected + upgraded review
  3. 练习       – AI-generated translation drills + essay-outline drills
  4. 设置       – API config, connection test, library build

All AI calls run on background QThreads so the UI never blocks.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ai_client import AIClient, AIConfig, AIError, CONFIG_PATH, _log
from reviewer import load_library, review, render_markdown
from exercise_generator import (
    gen_translation,
    review_translation_attempt,
    gen_essay_topic,
    review_outline,
    render_translation_questions,
    render_translation_keys,
    render_essay_topic,
    render_outline_review,
    list_categories,
)

ROOT = Path(__file__).parent
REVIEWS_DIR = ROOT / "reviews"
REVIEWS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Background worker
# =============================================================================

class Worker(QObject):
    """Runs a single callable on a QThread and emits its result."""

    finished = Signal(object)   # result
    failed = Signal(str)        # error message
    progress = Signal(str)      # human-readable status updates

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self):
        fn_name = getattr(self._fn, "__name__", repr(self._fn))
        _log(f"Worker.run start: {fn_name}")
        try:
            result = self._fn(*self._args, **self._kwargs)
            _log(f"Worker.run done : {fn_name}")
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001
            _log(f"Worker.run fail : {fn_name}\n{traceback.format_exc()}")
            self.failed.emit(str(e) or type(e).__name__)


class _Dispatcher(QObject):
    """Lives in the main thread (parented to a main-thread QObject).

    Worker signals are connected to *slots on this object*, so Qt sees a clear
    cross-thread signal/slot pair and automatically uses QueuedConnection,
    which guarantees ``on_done`` / ``on_fail`` execute on the main thread —
    safe to touch GUI objects.
    """

    def __init__(self, on_done, on_fail, parent):
        super().__init__(parent)
        self._on_done = on_done
        self._on_fail = on_fail

    @Slot(object)
    def handle_done(self, result):
        try:
            self._on_done(result)
        except Exception as e:  # noqa: BLE001
            _log(f"Dispatcher.handle_done error: {type(e).__name__}: {e}")
            traceback.print_exc()

    @Slot(str)
    def handle_fail(self, err):
        try:
            if self._on_fail:
                self._on_fail(err)
        except Exception as e:  # noqa: BLE001
            _log(f"Dispatcher.handle_fail error: {type(e).__name__}: {e}")
            traceback.print_exc()


def run_in_thread(parent, fn, on_done, on_fail=None, on_progress=None, *args, **kwargs):
    """Spawn ``fn`` on a QThread and wire callbacks. Returns (thread, worker).

    Two PySide6 traps handled here:
      1. The Worker and the Dispatcher must be kept alive by Python until
         the thread finishes — otherwise GC silently kills the slot.
         We anchor them on the thread object.
      2. Cross-thread callbacks must land on the main thread. Connecting to a
         slot on a main-thread QObject (the Dispatcher) is the only reliably
         portable way to get a QueuedConnection auto-applied in PySide6.
    """
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    dispatcher = _Dispatcher(on_done, on_fail, parent)
    thread._py_worker = worker      # type: ignore[attr-defined]  # GC anchor
    thread._py_dispatcher = dispatcher  # type: ignore[attr-defined]
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(dispatcher.handle_done)  # main-thread slot
    worker.failed.connect(dispatcher.handle_fail)    # main-thread slot
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(dispatcher.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker


class ElapsedTicker:
    """Updates a status QLabel with elapsed time while a task runs."""

    def __init__(self, label, prefix: str):
        self.label = label
        self.prefix = prefix
        self.start_time = 0.0
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._tick)

    def start(self):
        import time as _t
        self.start_time = _t.time()
        self._tick()
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def _tick(self):
        import time as _t
        elapsed = int(_t.time() - self.start_time)
        self.label.setText(f"{self.prefix}（已等待 {elapsed} 秒）")


# =============================================================================
# Tab 1: Library browser
# =============================================================================

class LibraryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.library: list[dict] = []
        self._build_ui()
        self.reload_library()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("类别："))
        self.cat_combo = QComboBox()
        self.cat_combo.currentTextChanged.connect(self.apply_filter)
        controls.addWidget(self.cat_combo)

        controls.addWidget(QLabel("类型："))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["全部", "单词", "短语 / 词块", "例句"])
        self.type_combo.currentTextChanged.connect(self.apply_filter)
        controls.addWidget(self.type_combo)

        controls.addWidget(QLabel("搜索："))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("英文 / 中文 / 小节关键词")
        self.search_edit.textChanged.connect(self.apply_filter)
        controls.addWidget(self.search_edit, stretch=1)

        self.reload_btn = QPushButton("重新载入")
        self.reload_btn.clicked.connect(self.reload_library)
        controls.addWidget(self.reload_btn)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["类别", "小节", "英文", "中文", "类型"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        self.count_label = QLabel("0 条")
        layout.addWidget(self.count_label)

    def reload_library(self):
        self.library = load_library()
        cats = sorted({e.get("category", "") for e in self.library if e.get("category")})
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        self.cat_combo.addItem("全部")
        self.cat_combo.addItems(cats)
        self.cat_combo.blockSignals(False)
        self.apply_filter()

    def apply_filter(self):
        cat = self.cat_combo.currentText()
        typ = self.type_combo.currentText()
        kw = self.search_edit.text().strip().lower()

        rows: list[dict] = []
        for e in self.library:
            if cat and cat != "全部" and e.get("category") != cat:
                continue
            if typ == "单词" and e.get("is_phrase"):
                continue
            if typ == "短语 / 词块" and not e.get("is_phrase"):
                continue
            if typ == "例句" and not e.get("is_example"):
                continue
            if kw:
                blob = (
                    f"{e.get('english','')} {e.get('chinese','')} "
                    f"{e.get('section','')} {e.get('category','')}"
                ).lower()
                if kw not in blob:
                    continue
            rows.append(e)

        self.table.setRowCount(len(rows))
        for i, e in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(e.get("category", "")))
            self.table.setItem(i, 1, QTableWidgetItem(e.get("section", "")))
            self.table.setItem(i, 2, QTableWidgetItem(e.get("english", "")))
            self.table.setItem(i, 3, QTableWidgetItem(e.get("chinese", "")))
            tag = "例句" if e.get("is_example") else ("短语" if e.get("is_phrase") else "单词")
            self.table.setItem(i, 4, QTableWidgetItem(tag))
        self.count_label.setText(f"{len(rows)} 条 / 共 {len(self.library)} 条")


# =============================================================================
# Tab 2: Review
# =============================================================================

class ReviewTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._thread = None
        self._ticker = ElapsedTicker(self.status_label, "⏳ AI 批改中")

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ---- left: input ----
        left = QWidget()
        ll = QVBoxLayout(left)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("批改类型："))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["作文 (essay)", "中译英翻译 (translation)"])
        type_row.addWidget(self.type_combo)
        type_row.addStretch()
        ll.addLayout(type_row)

        ll.addWidget(QLabel("题目 / 中文原文（可选）："))
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "如果是作文，粘贴作文题目；如果是翻译，粘贴中文原文。"
        )
        self.prompt_edit.setMaximumHeight(120)
        ll.addWidget(self.prompt_edit)

        ll.addWidget(QLabel("你的英文提交："))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("在此粘贴你的英文作文 / 翻译")
        ll.addWidget(self.text_edit, stretch=1)

        btn_row = QHBoxLayout()
        self.submit_btn = QPushButton("开始批改")
        self.submit_btn.clicked.connect(self.do_review)
        btn_row.addWidget(self.submit_btn)
        self.load_btn = QPushButton("从文件载入...")
        self.load_btn.clicked.connect(self.load_file)
        btn_row.addWidget(self.load_btn)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_inputs)
        btn_row.addWidget(self.clear_btn)
        ll.addLayout(btn_row)

        splitter.addWidget(left)

        # ---- right: result ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("批改结果："))
        self.result_view = QTextBrowser()
        self.result_view.setOpenExternalLinks(True)
        rl.addWidget(self.result_view, stretch=1)
        bottom_row = QHBoxLayout()
        self.status_label = QLabel("准备就绪")
        bottom_row.addWidget(self.status_label, stretch=1)
        self.save_btn = QPushButton("保存批改报告...")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        bottom_row.addWidget(self.save_btn)
        rl.addLayout(bottom_row)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self._last_md = ""

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "文本文件 (*.txt *.md);;所有文件 (*)"
        )
        if path:
            try:
                self.text_edit.setPlainText(
                    Path(path).read_text(encoding="utf-8")
                )
            except Exception as e:
                QMessageBox.warning(self, "载入失败", str(e))

    def clear_inputs(self):
        self.prompt_edit.clear()
        self.text_edit.clear()
        self.result_view.clear()
        self.status_label.setText("准备就绪")
        self.save_btn.setEnabled(False)

    def do_review(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "无内容", "请先粘贴英文作文或译文。")
            return
        task_type = "essay" if self.type_combo.currentIndex() == 0 else "translation"
        prompt = self.prompt_edit.toPlainText().strip()

        self.submit_btn.setEnabled(False)
        self._ticker.start()
        self.result_view.setMarkdown("_正在批改，请稍候..._")

        def task():
            return review(text, task_type=task_type, prompt_text=prompt)

        def on_done(result):
            self._ticker.stop()
            md = render_markdown(
                result, task_type=task_type, prompt_text=prompt, student_text=text
            )
            self._last_md = md
            self.result_view.setMarkdown(md)
            self.status_label.setText("✓ 批改完成")
            self.submit_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

        def on_fail(err):
            self._ticker.stop()
            self.result_view.setMarkdown(f"## 批改失败\n\n```\n{err}\n```")
            self.status_label.setText("✗ 批改失败")
            self.submit_btn.setEnabled(True)

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)

    def save_result(self):
        if not self._last_md:
            return
        default = REVIEWS_DIR / f"review_{datetime.now():%Y%m%d_%H%M%S}.md"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存批改报告", str(default), "Markdown (*.md)"
        )
        if path:
            Path(path).write_text(self._last_md, encoding="utf-8")
            self.status_label.setText(f"已保存：{Path(path).name}")


# =============================================================================
# Tab 3: Exercises (translation drill + essay outline drill)
# =============================================================================

class TranslationExerciseWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._items: list[dict] = []
        self._show_hints: bool = False
        self._show_keys: bool = False
        self._thread = None
        self._build_ui()
        self._gen_ticker = ElapsedTicker(self.status_label, "⏳ 出题中")
        self._sub_ticker = ElapsedTicker(self.status_label, "⏳ 批改中")

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- top controls ----
        controls = QHBoxLayout()
        controls.addWidget(QLabel("题量："))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(5)
        controls.addWidget(self.n_spin)
        controls.addWidget(QLabel("类别："))
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("（任意）")
        controls.addWidget(self.cat_combo)
        self.gen_btn = QPushButton("出题")
        self.gen_btn.clicked.connect(self.do_generate)
        controls.addWidget(self.gen_btn)
        self.hint_btn = QPushButton("显示提示")
        self.hint_btn.setEnabled(False)
        self.hint_btn.clicked.connect(self.toggle_hints)
        controls.addWidget(self.hint_btn)
        self.key_btn = QPushButton("显示参考答案")
        self.key_btn.setEnabled(False)
        self.key_btn.clicked.connect(self.toggle_keys)
        controls.addWidget(self.key_btn)
        controls.addStretch()
        self.status_label = QLabel("准备就绪")
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        # ---- vertical splitter: questions / answers / result ----
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        # 1) questions display
        q_box = QWidget()
        qv = QVBoxLayout(q_box)
        qv.addWidget(QLabel("题目："))
        self.q_view = QTextBrowser()
        self.q_view.setPlaceholderText(
            "点击「出题」生成翻译练习。点「显示提示」可显示每题建议使用的高级表达。"
        )
        qv.addWidget(self.q_view, stretch=1)
        splitter.addWidget(q_box)

        # 2) single answer textarea
        a_box = QWidget()
        av = QVBoxLayout(a_box)
        av.addWidget(QLabel("你的英文译文（每题前用 1. / 2. / 3. 标号；可以只答其中几题）："))
        self.answer_edit = QPlainTextEdit()
        self.answer_edit.setPlaceholderText(
            "1. Your translation for question 1...\n2. Your translation for question 2...\n3. ..."
        )
        av.addWidget(self.answer_edit, stretch=1)
        bbtn = QHBoxLayout()
        self.submit_btn = QPushButton("提交批改")
        self.submit_btn.setEnabled(False)
        self.submit_btn.clicked.connect(self.do_submit)
        bbtn.addWidget(self.submit_btn)
        bbtn.addStretch()
        av.addLayout(bbtn)
        splitter.addWidget(a_box)

        # 3) result area
        r_box = QWidget()
        rv = QVBoxLayout(r_box)
        rv.addWidget(QLabel("批改结果 / 参考答案："))
        self.result_view = QTextBrowser()
        rv.addWidget(self.result_view, stretch=1)
        splitter.addWidget(r_box)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def populate_categories(self, library: list[dict]):
        cats = sorted({e.get("category", "") for e in library if e.get("category")})
        self.cat_combo.clear()
        self.cat_combo.addItem("（任意）")
        self.cat_combo.addItems(cats)

    def _render_questions(self):
        if not self._items:
            self.q_view.clear()
            return
        lines: list[str] = []
        for i, item in enumerate(self._items, start=1):
            lines.append(f"### 第 {i} 题")
            lines.append("")
            lines.append(f"**中文**：{item.get('chinese','')}")
            if self._show_hints:
                tgt = item.get("target_expressions") or []
                if tgt:
                    lines.append("")
                    lines.append(f"_💡 建议使用：{', '.join(tgt)}_")
            if self._show_keys:
                ref = item.get("reference_translation", "")
                if ref:
                    lines.append("")
                    lines.append(f"_✅ 参考译文：{ref}_")
                pts = item.get("key_points") or []
                if pts:
                    lines.append(f"_得分点：{', '.join(pts)}_")
            lines.append("")
        self.q_view.setMarkdown("\n".join(lines))

    def toggle_hints(self):
        self._show_hints = not self._show_hints
        self.hint_btn.setText("隐藏提示" if self._show_hints else "显示提示")
        self._render_questions()

    def toggle_keys(self):
        self._show_keys = not self._show_keys
        self.key_btn.setText(
            "隐藏参考答案" if self._show_keys else "显示参考答案"
        )
        self._render_questions()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def do_generate(self):
        self.gen_btn.setEnabled(False)
        self.submit_btn.setEnabled(False)
        self.key_btn.setEnabled(False)
        self.hint_btn.setEnabled(False)
        self._gen_ticker.start()
        self.result_view.clear()
        self.answer_edit.clear()

        n = self.n_spin.value()
        cat = self.cat_combo.currentText()
        cat = None if cat == "（任意）" else cat

        def task():
            return gen_translation(n=n, category_filter=cat)

        def on_done(data):
            self._gen_ticker.stop()
            self._items = data.get("items", [])
            # Reset toggles on each new generation
            self._show_hints = False
            self._show_keys = False
            self.hint_btn.setText("显示提示")
            self.key_btn.setText("显示参考答案")
            self._render_questions()
            self.submit_btn.setEnabled(bool(self._items))
            self.key_btn.setEnabled(bool(self._items))
            self.hint_btn.setEnabled(bool(self._items))
            self.gen_btn.setEnabled(True)
            self.status_label.setText(f"✓ 已生成 {len(self._items)} 题")

        def on_fail(err):
            self._gen_ticker.stop()
            QMessageBox.warning(self, "出题失败", err)
            self.gen_btn.setEnabled(True)
            self.status_label.setText("✗ 出题失败")

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_answers(raw_text: str, n: int) -> list[str]:
        """Parse user-written ``1. xxx\\n2. xxx\\n...`` into n answer strings.

        Lines not preceded by a number prefix are appended to the current item.
        Returns a list of length n; missing items are ''.
        """
        import re

        answers: list[str] = [""] * n
        cur_idx: int = -1
        buf: list[str] = []
        for line in raw_text.splitlines():
            m = re.match(r"^\s*(\d+)\s*[\.、\)\）]\s*(.*)$", line)
            if m:
                if 0 <= cur_idx < n:
                    answers[cur_idx] = "\n".join(buf).strip()
                idx = int(m.group(1)) - 1
                cur_idx = idx
                buf = [m.group(2)]
            else:
                buf.append(line)
        if 0 <= cur_idx < n:
            answers[cur_idx] = "\n".join(buf).strip()
        return answers

    def do_submit(self):
        if not self._items:
            return
        raw = self.answer_edit.toPlainText()
        answers = self._parse_answers(raw, len(self._items))
        if not any(a.strip() for a in answers):
            QMessageBox.information(
                self,
                "未识别到答案",
                "请用 1. / 2. / 3. 这样的标号标记每题答案，至少填写一题再提交。",
            )
            return

        self.submit_btn.setEnabled(False)
        self._sub_ticker.start()
        self.result_view.setMarkdown("_正在批改你的译文，请稍候..._")

        items_copy = list(self._items)

        def task():
            outputs = []
            for item, ans in zip(items_copy, answers):
                if not ans:
                    outputs.append(None)
                    continue
                outputs.append(review_translation_attempt(item, ans))
            return outputs

        def on_done(results):
            md_parts = ["# 翻译批改结果", ""]
            for i, (item, ans, res) in enumerate(
                zip(items_copy, answers, results), start=1
            ):
                md_parts.append(f"## 第 {i} 题")
                md_parts.append("")
                md_parts.append(f"**中文**：{item.get('chinese','')}")
                md_parts.append("")
                md_parts.append(f"**你的译文**：{ans or '（未作答）'}")
                md_parts.append("")
                md_parts.append(
                    f"**参考译文**：{item.get('reference_translation','')}"
                )
                md_parts.append("")
                if res is None:
                    md_parts.append("_（你未作答，跳过批改）_")
                    md_parts.append("")
                    continue
                ov = res.get("overall") or {}
                md_parts.append(f"**评分档**：{ov.get('band','—')}")
                if ov.get("highlight"):
                    md_parts.append(f"**亮点**：{ov['highlight']}")
                if ov.get("to_improve"):
                    md_parts.append(f"**改进**：{ov['to_improve']}")
                md_parts.append("")
                corrs = res.get("corrections") or []
                if corrs:
                    md_parts.append("**错误纠正**：")
                    for c in corrs:
                        md_parts.append(
                            f"- [{c.get('type','')}] {c.get('original','')} → "
                            f"{c.get('fixed','')}"
                        )
                    md_parts.append("")
                ups = res.get("upgrades") or []
                if ups:
                    md_parts.append("**学习库升级建议**：")
                    for u in ups:
                        md_parts.append(
                            f"- {u.get('original','')} → "
                            f"**{u.get('suggested','')}**（{u.get('reason','')}）"
                        )
                    md_parts.append("")
            self._sub_ticker.stop()
            self.result_view.setMarkdown("\n".join(md_parts))
            self.submit_btn.setEnabled(True)
            self.status_label.setText("✓ 批改完成")

        def on_fail(err):
            self._sub_ticker.stop()
            self.result_view.setMarkdown(f"## 批改失败\n\n```\n{err}\n```")
            self.submit_btn.setEnabled(True)
            self.status_label.setText("✗ 批改失败")

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)


class EssayOutlineWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._topic: dict | None = None
        self._thread = None
        self._build_ui()
        self._gen_ticker = ElapsedTicker(self.status_label, "⏳ 出题中")
        self._sub_ticker = ElapsedTicker(self.status_label, "⏳ 批改中")

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("话题（可选）："))
        self.topic_combo = QComboBox()
        self.topic_combo.addItem("（AI 随机挑选）")
        controls.addWidget(self.topic_combo)
        self.gen_btn = QPushButton("出题")
        self.gen_btn.clicked.connect(self.do_generate)
        controls.addWidget(self.gen_btn)
        controls.addStretch()
        self.status_label = QLabel("准备就绪")
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.addWidget(QLabel("作文题："))
        self.topic_view = QTextBrowser()
        tl.addWidget(self.topic_view, stretch=1)
        splitter.addWidget(top)

        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.addWidget(QLabel("你的大纲（中文 / 英文均可，按思路写要点）："))
        self.outline_edit = QPlainTextEdit()
        self.outline_edit.setPlaceholderText(
            "例：\n开头：引出现象 / 阐明立场...\n论点1：...\n论点2：...\n结尾：..."
        )
        ml.addWidget(self.outline_edit, stretch=1)
        self.submit_btn = QPushButton("提交批改")
        self.submit_btn.setEnabled(False)
        self.submit_btn.clicked.connect(self.do_submit)
        ml.addWidget(self.submit_btn)
        splitter.addWidget(mid)

        bot = QWidget()
        bl = QVBoxLayout(bot)
        bl.addWidget(QLabel("批改结果："))
        self.result_view = QTextBrowser()
        bl.addWidget(self.result_view, stretch=1)
        splitter.addWidget(bot)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

    def populate_categories(self, library: list[dict]):
        cats = sorted({e.get("category", "") for e in library if e.get("category")})
        self.topic_combo.clear()
        self.topic_combo.addItem("（AI 随机挑选）")
        # Only suggest thematic categories (not 高级XXX合集)
        thematic = [c for c in cats if "合集" not in c and "作文" not in c]
        self.topic_combo.addItems(thematic)

    def do_generate(self):
        self.gen_btn.setEnabled(False)
        self.submit_btn.setEnabled(False)
        self._gen_ticker.start()
        self.topic_view.clear()
        self.result_view.clear()

        hint = self.topic_combo.currentText()
        hint = None if hint == "（AI 随机挑选）" else hint

        def task():
            return gen_essay_topic(topic_hint=hint)

        def on_done(topic):
            self._gen_ticker.stop()
            self._topic = topic
            self.topic_view.setMarkdown(render_essay_topic(topic))
            self.submit_btn.setEnabled(True)
            self.gen_btn.setEnabled(True)
            self.status_label.setText("✓ 题目已生成")

        def on_fail(err):
            self._gen_ticker.stop()
            QMessageBox.warning(self, "出题失败", err)
            self.gen_btn.setEnabled(True)
            self.status_label.setText("✗ 出题失败")

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)

    def do_submit(self):
        if not self._topic:
            return
        outline = self.outline_edit.toPlainText().strip()
        if not outline:
            QMessageBox.information(self, "未填写", "请先写大纲。")
            return
        self.submit_btn.setEnabled(False)
        self._sub_ticker.start()
        self.result_view.setMarkdown("_正在批改大纲，请稍候..._")

        topic = self._topic

        def task():
            return review_outline(topic, outline)

        def on_done(res):
            self._sub_ticker.stop()
            self.result_view.setMarkdown(render_outline_review(res))
            self.submit_btn.setEnabled(True)
            self.status_label.setText("✓ 批改完成")

        def on_fail(err):
            self._sub_ticker.stop()
            self.result_view.setMarkdown(f"## 批改失败\n\n```\n{err}\n```")
            self.submit_btn.setEnabled(True)
            self.status_label.setText("✗ 批改失败")

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)


class SpotCheckTab(QWidget):
    """Local flashcard-style spot check — draws random entries straight from the
    library (no AI call). Reveal answers / re-draw on demand."""

    def __init__(self):
        super().__init__()
        self._library: list[dict] = []
        self._items: list[dict] = []
        self._show_answer: bool = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("方向："))
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["看中文回想英文", "看英文回想中文"])
        self.dir_combo.currentTextChanged.connect(self._render)
        controls.addWidget(self.dir_combo)

        controls.addWidget(QLabel("类型："))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["全部", "单词", "短语 / 词块"])
        controls.addWidget(self.type_combo)

        controls.addWidget(QLabel("类别："))
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("（任意）")
        controls.addWidget(self.cat_combo)

        controls.addWidget(QLabel("数量："))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 20)
        self.n_spin.setValue(8)
        controls.addWidget(self.n_spin)

        self.draw_btn = QPushButton("抽取")
        self.draw_btn.clicked.connect(self.do_draw)
        controls.addWidget(self.draw_btn)

        self.redraw_btn = QPushButton("重新抽取")
        self.redraw_btn.setEnabled(False)
        self.redraw_btn.clicked.connect(self.do_draw)
        controls.addWidget(self.redraw_btn)

        self.answer_btn = QPushButton("显示答案")
        self.answer_btn.setEnabled(False)
        self.answer_btn.clicked.connect(self.toggle_answer)
        controls.addWidget(self.answer_btn)

        controls.addStretch()
        self.status_label = QLabel("准备就绪")
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        self.view = QTextBrowser()
        self.view.setPlaceholderText(
            "选择筛选条件后点「抽取」，会从学习库随机抽取若干条目进行自测。"
        )
        layout.addWidget(self.view, stretch=1)

    def reload_library(self):
        self._library = load_library()
        cats = sorted({e.get("category", "") for e in self._library if e.get("category")})
        self.cat_combo.clear()
        self.cat_combo.addItem("（任意）")
        self.cat_combo.addItems(cats)

    def do_draw(self):
        import random

        if not self._library:
            self.reload_library()
        if not self._library:
            QMessageBox.information(
                self, "学习库为空", "请先在「设置」里构建学习库。"
            )
            return

        typ = self.type_combo.currentText()
        cat = self.cat_combo.currentText()
        pool = self._library
        if cat != "（任意）":
            pool = [e for e in pool if e.get("category") == cat]
        if typ == "单词":
            pool = [e for e in pool if not e.get("is_phrase")]
        elif typ == "短语 / 词块":
            pool = [e for e in pool if e.get("is_phrase")]

        if not pool:
            QMessageBox.information(
                self, "无可抽取条目", "当前筛选条件下没有条目，换个类型 / 类别试试。"
            )
            return

        n = min(self.n_spin.value(), len(pool))
        self._items = random.sample(pool, n)
        self._show_answer = False
        self.answer_btn.setText("显示答案")
        self.answer_btn.setEnabled(True)
        self.redraw_btn.setEnabled(True)
        self._render()
        self.status_label.setText(f"已抽取 {n} 条（共 {len(pool)} 条可选）")

    def toggle_answer(self):
        self._show_answer = not self._show_answer
        self.answer_btn.setText("隐藏答案" if self._show_answer else "显示答案")
        self._render()

    def _render(self):
        if not self._items:
            return
        cn_to_en = self.dir_combo.currentIndex() == 0
        lines: list[str] = []
        for i, e in enumerate(self._items, start=1):
            tag = "短语" if e.get("is_phrase") else "单词"
            cat = e.get("category", "")
            lines.append(f"#### {i}. （{cat} · {tag}）")
            lines.append("")
            if cn_to_en:
                lines.append(f"**中文**：{e.get('chinese','')}")
                if self._show_answer:
                    lines.append(f"**英文**：{e.get('english','')}")
                else:
                    lines.append("**英文**：　＿＿＿＿＿＿＿＿＿＿")
            else:
                lines.append(f"**英文**：{e.get('english','')}")
                if self._show_answer:
                    lines.append(f"**中文**：{e.get('chinese','')}")
                else:
                    lines.append("**中文**：　＿＿＿＿＿＿＿＿＿＿")
            lines.append("")
        self.view.setMarkdown("\n".join(lines))


class ExerciseTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.inner = QTabWidget()
        layout.addWidget(self.inner)
        self.trans_widget = TranslationExerciseWidget()
        self.essay_widget = EssayOutlineWidget()
        self.inner.addTab(self.trans_widget, "翻译练习")
        self.inner.addTab(self.essay_widget, "作文大纲练习")

    def reload_library(self):
        lib = load_library()
        self.trans_widget.populate_categories(lib)
        self.essay_widget.populate_categories(lib)


# =============================================================================
# Tab 4: Settings
# =============================================================================

class SettingsTab(QWidget):
    library_built = Signal()

    def __init__(self):
        super().__init__()
        self._thread = None
        self._build_ui()
        self.load_config()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        api_box = QGroupBox("API 配置")
        form = QFormLayout(api_box)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API Key：", self.api_key_edit)
        self.base_url_edit = QLineEdit()
        form.addRow("Base URL：", self.base_url_edit)
        self.model_edit = QLineEdit()
        form.addRow("模型名：", self.model_edit)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        form.addRow("超时（秒）：", self.timeout_spin)
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 10)
        form.addRow("重试次数：", self.retries_spin)
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.save_config)
        btn_row.addWidget(self.save_btn)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        btn_row.addWidget(self.test_btn)
        btn_row.addStretch()
        form.addRow(btn_row)
        self.api_status = QLabel("")
        form.addRow(self.api_status)
        layout.addWidget(api_box)

        lib_box = QGroupBox("学习库")
        lv = QVBoxLayout(lib_box)
        info = QLabel(
            "首次使用需构建学习库。由于使用 thinking 模型，全量构建大约需要 2 小时。"
            "构建过程可中断，下次运行会从缓存继续。"
        )
        info.setWordWrap(True)
        lv.addWidget(info)

        b_row = QHBoxLayout()
        self.build_btn = QPushButton("构建 / 继续构建学习库")
        self.build_btn.clicked.connect(lambda: self.build_library(refresh=False))
        b_row.addWidget(self.build_btn)
        self.refresh_btn = QPushButton("强制重新构建（忽略缓存）")
        self.refresh_btn.clicked.connect(lambda: self.build_library(refresh=True))
        b_row.addWidget(self.refresh_btn)
        b_row.addStretch()
        lv.addLayout(b_row)

        self.build_progress = QProgressBar()
        self.build_progress.setVisible(False)
        lv.addWidget(self.build_progress)
        self.build_log = QPlainTextEdit()
        self.build_log.setReadOnly(True)
        self.build_log.setMaximumHeight(220)
        lv.addWidget(self.build_log)
        layout.addWidget(lib_box)

        layout.addStretch()

    def load_config(self):
        try:
            cfg = AIConfig.load()
            self.api_key_edit.setText(cfg.api_key)
            self.base_url_edit.setText(cfg.base_url)
            self.model_edit.setText(cfg.model)
            self.timeout_spin.setValue(cfg.timeout)
            self.retries_spin.setValue(cfg.max_retries)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "配置读取失败", str(e))

    def save_config(self):
        cfg = AIConfig(
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip(),
            model=self.model_edit.text().strip(),
            timeout=self.timeout_spin.value(),
            max_retries=self.retries_spin.value(),
            backoff=2.0,
        )
        cfg.save()
        self.api_status.setText("✓ 已保存")

    def test_connection(self):
        self.test_btn.setEnabled(False)
        self.api_status.setText("⏳ 测试中...")
        # Re-load just in case the user typed but didn't save.
        cfg = AIConfig(
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip(),
            model=self.model_edit.text().strip(),
            timeout=self.timeout_spin.value(),
            max_retries=self.retries_spin.value(),
            backoff=2.0,
        )

        def task():
            return AIClient(cfg).ping()

        def on_done(result):
            ok, msg = result
            self.api_status.setText(("✓ " if ok else "✗ ") + msg)
            self.test_btn.setEnabled(True)

        def on_fail(err):
            self.api_status.setText("✗ " + err.splitlines()[0])
            self.test_btn.setEnabled(True)

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)

    def build_library(self, refresh: bool):
        from library_builder import build

        self.build_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.build_progress.setVisible(True)
        self.build_progress.setRange(0, 0)  # busy indicator
        self.build_log.clear()
        self.build_log.appendPlainText("开始构建学习库...")

        def task():
            # Note: heavy job — call directly here on the worker thread.
            return build(refresh=refresh, verbose=True)

        def on_done(stats):
            self.build_log.appendPlainText(
                f"\n=== 构建完成 ===\n"
                f"  in: {stats['in']}  kept: {stats['kept']}  "
                f"dropped: {stats['dropped']}"
            )
            self.build_progress.setVisible(False)
            self.build_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.library_built.emit()

        def on_fail(err):
            self.build_log.appendPlainText("\n=== 构建失败 ===\n" + err)
            self.build_progress.setVisible(False)
            self.build_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)

        self._thread, _ = run_in_thread(self, task, on_done, on_fail)


# =============================================================================
# Main window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("我的英语学习与批改助手 · CET-4/6")
        self.resize(1280, 820)

        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

        tabs = QTabWidget()
        self.library_tab = LibraryTab()
        self.spotcheck_tab = SpotCheckTab()
        self.review_tab = ReviewTab()
        self.exercise_tab = ExerciseTab()
        self.settings_tab = SettingsTab()
        tabs.addTab(self.library_tab, "学习库")
        tabs.addTab(self.spotcheck_tab, "抽查")
        tabs.addTab(self.review_tab, "批改")
        tabs.addTab(self.exercise_tab, "练习")
        tabs.addTab(self.settings_tab, "设置")
        self.setCentralWidget(tabs)

        sb = QStatusBar()
        cfg_path = CONFIG_PATH.name
        sb.showMessage(f"配置文件：{cfg_path}    |    Ctrl+, 切到「设置」管理 API")
        self.setStatusBar(sb)

        self.exercise_tab.reload_library()
        self.spotcheck_tab.reload_library()
        self.settings_tab.library_built.connect(self._after_library_built)

    def _after_library_built(self):
        self.library_tab.reload_library()
        self.exercise_tab.reload_library()
        self.spotcheck_tab.reload_library()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
