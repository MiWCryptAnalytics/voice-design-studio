"""Pronunciation window: respelling dictionary plus text normalization.

Two halves. The dictionary is user respellings — the only pronunciation lever
this model offers, since it takes no phoneme input. Normalization is the
automatic written-to-spoken pass. Order is fixed: normalize first, rules last,
so a hand-written rule can always override an automatic transform.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.normalize import (
    NUMERIC_STEPS,
    STEP_LABELS,
    NormalizationOptions,
    supports_numeric,
)
from ..core.pronounce import ANY_LANGUAGE, PronunciationBook, PronunciationRule
from . import theme
from .metrics import metrics
from .widgets import ThemedComboBox

# Kept short so Match and Say-as — the columns that actually hold content — get
# the width. Full meanings live in the header tooltips.
COLUMNS = ["On", "Match", "Say as", "Word", "Case", "Re", "Lang", "Hits"]
COLUMN_TOOLTIPS = [
    "Rule enabled",
    "Text to look for",
    "How it should be spoken",
    "Match whole words only",
    "Case sensitive",
    "Treat the match as a regular expression",
    "Language this rule applies to",
    "Times this rule matched the preview text",
]
CARRIER = "Here is the word {word} spoken in a sentence."


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    return label


class PronunciationWindow(QDialog):
    changed = pyqtSignal()
    testRequested = pyqtSignal(str, str)  # rule id, test phrase

    def __init__(
        self,
        book: PronunciationBook,
        languages: Callable[[], list[str]],
        current_language: Callable[[], str],
        sample_text: Callable[[], str],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.book = book
        self._languages = languages
        self._current_language = current_language
        self._sample_text = sample_text
        self._loading = False

        self.setWindowTitle("Pronunciation")
        # Non-modal: keep editing rules while the studio generates.
        self.setModal(False)
        m = metrics()
        self.resize(m.ch(120), m.sp(38))

        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        layout.addLayout(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_dictionary())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        self.reload()

    # ---------- construction ----------

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.45))
        self.enabled_check = QCheckBox("Apply pronunciation to generated speech")
        self.enabled_check.setToolTip(
            "Off passes your text to the model exactly as written."
        )
        self.enabled_check.toggled.connect(self._on_enabled_toggled)
        row.addWidget(self.enabled_check)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    def _build_dictionary(self) -> QWidget:
        m = metrics()
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, m.sp(0.35), 0)
        box.setSpacing(m.sp(0.35))

        box.addWidget(_title("DICTIONARY"))
        box.addWidget(_hint(
            "Respellings applied to the text before synthesis. This model takes no "
            "phonemes, so write how it should sound: Qwen → Chwen. Earlier rules "
            "run first and later rules see their output."
        ))

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for i, tip in enumerate(COLUMN_TOOLTIPS):
            self.table.horizontalHeaderItem(i).setToolTip(tip)
        header = self.table.horizontalHeader()
        for i in range(len(COLUMNS)):
            mode = (QHeaderView.ResizeMode.Stretch if i in (1, 2)
                    else QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(i, mode)
        header.setMinimumSectionSize(m.ch(4))
        self.table.itemChanged.connect(self._on_cell_changed)
        self.table.itemSelectionChanged.connect(self._update_test_phrase)
        box.addWidget(self.table, 1)

        box.addLayout(self._build_rule_buttons())
        box.addWidget(self._build_tuning())
        return panel

    def _build_rule_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.35))

        def button(text: str, handler, tip: str = "", role: str = "") -> QPushButton:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            if tip:
                btn.setToolTip(tip)
            if role:
                btn.setProperty("role", role)
            return btn

        row.addWidget(button("Add rule", self._add_rule))
        row.addWidget(button("▲", lambda: self._move(-1), "Run earlier"))
        row.addWidget(button("▼", lambda: self._move(+1), "Run later"))
        row.addStretch(1)
        row.addWidget(button("Delete", self._delete_rule, role="danger"))
        return row

    def _build_tuning(self) -> QWidget:
        m = metrics()
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, m.sp(0.35), 0, 0)
        box.setSpacing(m.sp(0.3))

        box.addWidget(_title("TUNING"))
        box.addWidget(_hint(
            "Hear the selected rule: generates this phrase with the rule applied "
            "and without it, then pins them as A and B for instant comparison."
        ))

        row = QHBoxLayout()
        row.setSpacing(m.sp(0.35))
        self.test_phrase = QLineEdit()
        self.test_phrase.setPlaceholderText("Phrase to test the selected rule with…")
        self.test_button = QPushButton("Test A/B")
        self.test_button.setProperty("role", "primary")
        self.test_button.clicked.connect(self._on_test)
        row.addWidget(self.test_phrase, 1)
        row.addWidget(self.test_button)
        box.addLayout(row)
        return panel

    def _build_right_column(self) -> QWidget:
        m = metrics()
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(m.sp(0.35), 0, 0, 0)
        box.setSpacing(m.sp(0.35))

        box.addWidget(_title("NORMALIZATION"))
        self.language_note = _hint("")
        box.addWidget(self.language_note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        options_box = QVBoxLayout(holder)
        options_box.setContentsMargins(0, 0, 0, 0)
        options_box.setSpacing(m.sp(0.25))

        self.option_checks: dict[str, QCheckBox] = {}
        for name, label in STEP_LABELS.items():
            check = QCheckBox(label)
            if name in NUMERIC_STEPS:
                check.setToolTip("English only — skipped for other languages.")
            check.toggled.connect(self._on_option_toggled)
            self.option_checks[name] = check
            options_box.addWidget(check)
        options_box.addStretch(1)
        scroll.setWidget(holder)
        box.addWidget(scroll, 1)

        box.addWidget(_title("PREVIEW"))
        box.addWidget(_hint("Your script, and what will actually be spoken."))

        self.preview_in = QPlainTextEdit()
        self.preview_in.setReadOnly(True)
        self.preview_in.setMinimumHeight(m.sp(3.5))
        self.preview_out = QPlainTextEdit()
        self.preview_out.setReadOnly(True)
        self.preview_out.setMinimumHeight(m.sp(3.5))
        box.addWidget(self.preview_in)
        box.addWidget(self.preview_out)

        self.preview_summary = _hint("")
        box.addWidget(self.preview_summary)
        return panel

    # ---------- data ----------

    def reload(self) -> None:
        self._loading = True
        self.enabled_check.setChecked(self.book.enabled)
        options = self.book.options.to_dict()
        for name, check in self.option_checks.items():
            check.setChecked(bool(options.get(name, False)))
        self._populate_table()
        self._loading = False
        self.refresh_preview()

    def _populate_table(self) -> None:
        rules = self.book.all()
        self.table.setRowCount(len(rules))
        for row, rule in enumerate(rules):
            self._set_check(row, 0, rule.enabled)
            self._set_text(row, 1, rule.match, rule)
            self._set_text(row, 2, rule.replacement, rule)
            self._set_check(row, 3, rule.whole_word)
            self._set_check(row, 4, rule.case_sensitive)
            self._set_check(row, 5, rule.regex)
            self._set_language(row, 6, rule)
            hits = QTableWidgetItem("")
            hits.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 7, hits)

            if not rule.valid:
                item = self.table.item(row, 1)
                item.setForeground(QColor(theme.BAD))
                item.setToolTip("Invalid regular expression — this rule is skipped.")

    def _set_check(self, row: int, col: int, value: bool) -> None:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
        self.table.setItem(row, col, item)

    def _set_text(self, row: int, col: int, value: str, rule: PronunciationRule) -> None:
        item = QTableWidgetItem(value)
        item.setData(Qt.ItemDataRole.UserRole, rule.id)
        self.table.setItem(row, col, item)

    def _set_language(self, row: int, col: int, rule: PronunciationRule) -> None:
        combo = ThemedComboBox()
        combo.addItem(ANY_LANGUAGE)
        combo.addItems([lang for lang in self._languages() if lang != "Auto"])
        combo.setCurrentText(rule.language)
        # Don't let the longest language name dictate the column width.
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(5)
        combo.currentTextChanged.connect(
            lambda text, rid=rule.id: self._set_rule_field(rid, "language", text)
        )
        self.table.setCellWidget(row, col, combo)

    def _rule_at(self, row: int) -> PronunciationRule | None:
        item = self.table.item(row, 1)
        return self.book.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _selected_rule(self) -> PronunciationRule | None:
        rows = self.table.selectionModel().selectedRows()
        return self._rule_at(rows[0].row()) if rows else None

    # ---------- edits ----------

    def _set_rule_field(self, rule_id: str, field: str, value) -> None:
        rule = self.book.get(rule_id)
        if rule is None or getattr(rule, field) == value:
            return
        setattr(rule, field, value)
        self.book.update(rule)
        self._emit_changed()

    def _on_cell_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        rule = self._rule_at(item.row())
        if rule is None:
            return
        column = item.column()
        if column == 1:
            rule.match = item.text()
        elif column == 2:
            rule.replacement = item.text()
        elif column in (0, 3, 4, 5):
            checked = item.checkState() == Qt.CheckState.Checked
            setattr(rule, {0: "enabled", 3: "whole_word", 4: "case_sensitive",
                           5: "regex"}[column], checked)
        else:
            return
        self.book.update(rule)

        if column in (1, 5):  # match text or regex flag can invalidate the pattern
            self._loading = True
            self._populate_table()
            self._loading = False
        self._emit_changed()

    def _add_rule(self) -> None:
        rule = self.book.add(PronunciationRule(match="", replacement=""))
        self._loading = True
        self._populate_table()
        self._loading = False
        row = len(self.book.all()) - 1
        self.table.selectRow(row)
        self.table.editItem(self.table.item(row, 1))
        self._emit_changed()

    def _delete_rule(self) -> None:
        rule = self._selected_rule()
        if rule is None:
            return
        self.book.remove(rule.id)
        self._loading = True
        self._populate_table()
        self._loading = False
        self._emit_changed()

    def _move(self, delta: int) -> None:
        rule = self._selected_rule()
        if rule is None:
            return
        self.book.move(rule.id, delta)
        self._loading = True
        self._populate_table()
        self._loading = False
        index = next((i for i, r in enumerate(self.book.all()) if r.id == rule.id), 0)
        self.table.selectRow(index)
        self._emit_changed()

    def _on_enabled_toggled(self, value: bool) -> None:
        if self._loading:
            return
        self.book.enabled = value
        self.book.save()
        self._emit_changed()

    def _on_option_toggled(self) -> None:
        if self._loading:
            return
        self.book.options = NormalizationOptions(
            **{name: check.isChecked() for name, check in self.option_checks.items()}
        )
        self.book.save()
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.refresh_preview()
        self.changed.emit()

    # ---------- preview & tuning ----------

    def refresh_preview(self) -> None:
        language = self._current_language()
        numeric_ok = supports_numeric(language)
        self.language_note.setText(
            f"Language: {language}. Numeric and date rules are English-only and are "
            + ("applied." if numeric_ok else "skipped for this language.")
        )
        for name, check in self.option_checks.items():
            check.setEnabled(numeric_ok or name not in NUMERIC_STEPS)

        source = self._sample_text() or ""
        spoken = self.book.apply(source, language)
        self.preview_in.setPlainText(source)
        self.preview_out.setPlainText(spoken.text)

        for row in range(self.table.rowCount()):
            rule = self._rule_at(row)
            item = self.table.item(row, 7)
            if rule is not None and item is not None:
                count = spoken.rule_hits.get(rule.id, 0)
                item.setText(str(count) if count else "–")

        steps = ", ".join(spoken.normalization_steps) or "none"
        self.preview_summary.setText(
            f"Normalization applied: {steps} · rule matches: {spoken.total_rule_hits}"
            + ("" if spoken.changed else " · text unchanged")
        )

    def _update_test_phrase(self) -> None:
        rule = self._selected_rule()
        if rule is None or not rule.match:
            return
        if not self.test_phrase.text().strip() or rule.match not in self.test_phrase.text():
            word = rule.match if not rule.regex else "the pattern"
            self.test_phrase.setText(CARRIER.format(word=word))

    def _on_test(self) -> None:
        rule = self._selected_rule()
        if rule is None:
            QMessageBox.information(self, "No rule selected",
                                    "Select a rule in the table to test it.")
            return
        phrase = self.test_phrase.text().strip()
        if not phrase:
            QMessageBox.information(self, "No phrase",
                                    "Enter a phrase that contains the word to test.")
            return

        language = self._current_language()
        with_rule = self.book.apply(phrase, language)
        without_rule = self.book.apply(phrase, language, skip_rule=rule.id)
        if with_rule.text == without_rule.text:
            QMessageBox.information(
                self,
                "Rule doesn't change this phrase",
                f"“{rule.match}” didn't match anything in the test phrase, so both "
                "takes would be identical. Adjust the phrase or the rule.",
            )
            return
        self.testRequested.emit(rule.id, phrase)

    def set_test_busy(self, busy: bool) -> None:
        self.test_button.setEnabled(not busy)
