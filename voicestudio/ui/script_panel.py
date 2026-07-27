"""Center column top: the script to speak, language, and generate controls."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import dialogue as dialogue_mod
from ..core.script import split_script
from .metrics import metrics


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


class ScriptPanel(QFrame):
    generateRequested = pyqtSignal()
    variationsRequested = pyqtSignal(int)
    cancelRequested = pyqtSignal()
    dialogueDetected = pyqtSignal(object)  # DialogueScript, or None when plain prose

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        header.addWidget(_title("SCRIPT"))
        header.addStretch(1)
        self.stats_label = QLabel("")
        self.stats_label.setProperty("role", "hint")
        header.addWidget(self.stats_label)
        layout.addLayout(header)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("What should this voice say?")
        self.text_edit.setMinimumHeight(metrics().sp(6.5))
        self.text_edit.textChanged.connect(self._update_stats)
        layout.addWidget(self.text_edit, 1)

        layout.addLayout(self._build_options_row())
        layout.addLayout(self._build_action_row())
        self._update_stats()

    def _build_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.45))

        lang_label = QLabel("Language")
        lang_label.setProperty("role", "hint")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Auto", "English"])
        self.language_combo.setMinimumWidth(metrics().ch(11))

        self.split_check = QCheckBox("Split long scripts")
        self.split_check.setToolTip(
            "Break the script into chunks on sentence boundaries, generate each\n"
            "under the same seed, and join them into one take."
        )
        self.split_check.setChecked(True)
        self.split_check.toggled.connect(self._update_stats)

        row.addWidget(lang_label)
        row.addWidget(self.language_combo)
        row.addSpacing(metrics().sp(0.45))
        row.addWidget(self.split_check)
        row.addStretch(1)
        return row

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.45))

        self.generate_button = QPushButton("Generate")
        self.generate_button.setProperty("role", "primary")
        self.generate_button.setToolTip("Generate one take (Ctrl+Enter)")
        self.generate_button.clicked.connect(self.generateRequested.emit)

        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(2, 8)
        self.variations_spin.setValue(4)
        self.variations_spin.setToolTip("How many seeds to explore")

        self.variations_button = QPushButton("Generate ×N")
        self.variations_button.setToolTip(
            "Generate N takes from the same description with different seeds"
        )
        self.variations_button.clicked.connect(
            lambda: self.variationsRequested.emit(self.variations_spin.value())
        )

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.cancel_button.setVisible(False)

        row.addWidget(self.generate_button, 2)
        row.addWidget(self.variations_button, 1)
        row.addWidget(self.variations_spin)
        row.addWidget(self.cancel_button)
        return row

    # ---------- state ----------

    @property
    def script(self) -> str:
        return self.text_edit.toPlainText().strip()

    @property
    def language(self) -> str:
        return self.language_combo.currentText()

    @property
    def split_enabled(self) -> bool:
        return self.split_check.isChecked()

    def chunks(self) -> list[str]:
        text = self.script
        if not text:
            return []
        return split_script(text) if self.split_enabled else [text]

    def dialogue_script(self):
        """The parsed script when it uses speaker labels, else None."""
        text = self.script
        if not text:
            return None
        parsed = dialogue_mod.parse(text)
        return parsed if parsed.is_dialogue else None

    def set_languages(self, languages: list[str]) -> None:
        current = self.language_combo.currentText()
        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        self.language_combo.addItems(languages)
        if current in languages:
            self.language_combo.setCurrentText(current)
        self.language_combo.blockSignals(False)

    def set_busy(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy)
        self.variations_button.setEnabled(not busy)
        self.variations_spin.setEnabled(not busy)
        self.cancel_button.setVisible(busy)

    def _update_stats(self) -> None:
        text = self.script
        if not text:
            self.stats_label.setText("empty")
            self.dialogueDetected.emit(None)
            return

        n = len(text)
        parsed = self.dialogue_script()
        if parsed is not None:
            speakers = len(parsed.speakers)
            self.stats_label.setText(
                f"{n} chars · {len(parsed.lines)} lines · {speakers} speakers"
            )
            self.generate_button.setText("Generate dialogue")
        else:
            parts = self.chunks()
            self.stats_label.setText(
                f"{n} chars · {len(parts)} chunks" if len(parts) > 1 else f"{n} chars"
            )
            self.generate_button.setText("Generate")
        self.dialogueDetected.emit(parsed)
