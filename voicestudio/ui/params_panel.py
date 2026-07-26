"""Sampling parameters and seed control."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..engine import new_seed
from .metrics import metrics

DEFAULTS = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_new_tokens": 4096,
}


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    return label


class ParamsPanel(QFrame):
    seedChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")

        m = metrics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(0.7), m.sp(0.6), m.sp(0.7), m.sp(0.6))
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        title = QLabel("GENERATION")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        reset = QPushButton("Reset")
        reset.setProperty("role", "icon")
        reset.clicked.connect(self.reset_defaults)
        header.addWidget(reset)
        layout.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(m.sp(0.6))
        grid.setVerticalSpacing(m.sp(0.35))

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.temperature.setToolTip("Higher = more variation between takes")

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.05, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setDecimals(2)

        self.top_k = QSpinBox()
        self.top_k.setRange(0, 200)

        self.repetition_penalty = QDoubleSpinBox()
        self.repetition_penalty.setRange(1.0, 2.0)
        self.repetition_penalty.setSingleStep(0.01)
        self.repetition_penalty.setDecimals(2)

        self.max_new_tokens = QSpinBox()
        self.max_new_tokens.setRange(256, 8192)
        self.max_new_tokens.setSingleStep(256)
        self.max_new_tokens.setToolTip("Upper bound on generated audio length")

        fields = [
            ("Temperature", self.temperature),
            ("Top-p", self.top_p),
            ("Top-k", self.top_k),
            ("Rep. penalty", self.repetition_penalty),
            ("Max tokens", self.max_new_tokens),
        ]
        for i, (label, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            cell = QVBoxLayout()
            cell.setSpacing(metrics().sp(0.12))
            cell.addWidget(_hint(label))
            cell.addWidget(widget)
            container = QWidget()
            container.setLayout(cell)
            grid.addWidget(container, row, col)

        layout.addLayout(grid)
        layout.addLayout(self._build_seed_row())
        self.reset_defaults()

    def _build_seed_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.35))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self.seedChanged.emit)

        self.seed_lock = QCheckBox("Lock")
        self.seed_lock.setToolTip(
            "Reuse this seed for every take, so a description edit is the only\n"
            "thing that changes between generations."
        )

        dice = QPushButton("🎲")
        dice.setProperty("role", "icon")
        dice.setToolTip("New random seed")
        dice.clicked.connect(lambda: self.seed_spin.setValue(new_seed()))

        row.addWidget(_hint("Seed"))
        row.addWidget(self.seed_spin, 1)
        row.addWidget(dice)
        row.addWidget(self.seed_lock)
        return row

    # ---------- state ----------

    @property
    def seed_locked(self) -> bool:
        return self.seed_lock.isChecked()

    def effective_seed(self) -> int:
        """The seed to use for the next take."""
        if self.seed_locked:
            return self.seed_spin.value()
        seed = new_seed()
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(seed)
        self.seed_spin.blockSignals(False)
        return seed

    def show_seed(self, seed: int) -> None:
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(seed)
        self.seed_spin.blockSignals(False)

    def values(self) -> dict:
        return {
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "top_k": self.top_k.value(),
            "repetition_penalty": self.repetition_penalty.value(),
            "max_new_tokens": self.max_new_tokens.value(),
        }

    def apply(self, values: dict) -> None:
        self.temperature.setValue(float(values.get("temperature", DEFAULTS["temperature"])))
        self.top_p.setValue(float(values.get("top_p", DEFAULTS["top_p"])))
        self.top_k.setValue(int(values.get("top_k", DEFAULTS["top_k"])))
        self.repetition_penalty.setValue(
            float(values.get("repetition_penalty", DEFAULTS["repetition_penalty"]))
        )
        self.max_new_tokens.setValue(
            int(values.get("max_new_tokens", DEFAULTS["max_new_tokens"]))
        )

    def reset_defaults(self) -> None:
        self.apply(DEFAULTS)
