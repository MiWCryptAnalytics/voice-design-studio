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
    # Matches the checkpoint's generate_config.
    "max_new_tokens": 8192,
    "subtalker_do_sample": True,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
    "subtalker_top_k": 50,
}

SUBTALKER_KEYS = (
    "subtalker_do_sample",
    "subtalker_temperature",
    "subtalker_top_p",
    "subtalker_top_k",
)

SUBTALKER_TOOLTIP = (
    "Second sampling stage: predicts the residual codebooks that carry acoustic\n"
    "detail, on top of the talker's token stream.\n\n"
    "Measured (scripts/subtalker_sweep.py): changing these produces a clearly\n"
    "different rendition, but sweeping 0.2–1.5 showed no consistent direction —\n"
    "brightness, noisiness and cross-seed spread all stayed within run-to-run\n"
    "variance. Treat it as a second exploration axis, not a quality dial."
)


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
        self.max_new_tokens.setRange(64, 16384)
        self.max_new_tokens.setSingleStep(256)
        self.max_new_tokens.setToolTip(
            "Hard ceiling on generated codec frames.\n"
            "The codec runs at ~12 frames per second of audio, so 8192 is about\n"
            "11 minutes — far above any single chunk. If a take does hit this\n"
            "ceiling it is cut off mid-word and gets flagged as truncated."
        )

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
        layout.addWidget(self._build_subtalker_section())
        layout.addLayout(self._build_seed_row())
        # Keep the mirrored values visibly in step while linked.
        for widget in (self.temperature, self.top_p, self.top_k):
            widget.valueChanged.connect(self._sync_if_linked)
        self.reset_defaults()

    def _sync_if_linked(self) -> None:
        if self.subtalker_link.isChecked():
            self._mirror_main_to_subtalker()

    def _build_subtalker_section(self) -> QWidget:
        m = metrics()
        container = QWidget()
        box = QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(m.sp(0.3))

        header = QHBoxLayout()
        header.setSpacing(m.sp(0.35))
        self.subtalker_toggle = QPushButton("▸ Sub-talker")
        self.subtalker_toggle.setCheckable(True)
        self.subtalker_toggle.setToolTip(SUBTALKER_TOOLTIP)
        self.subtalker_toggle.toggled.connect(self._on_subtalker_toggled)

        self.subtalker_link = QCheckBox("Link to main")
        self.subtalker_link.setChecked(True)
        self.subtalker_link.setToolTip(
            "Mirror the values above, which is what the model does by default.\n"
            "Uncheck to sample the detail stage independently."
        )
        self.subtalker_link.toggled.connect(self._on_link_toggled)

        header.addWidget(self.subtalker_toggle, 1)
        header.addWidget(self.subtalker_link)
        box.addLayout(header)

        self.subtalker_body = QWidget()
        grid = QGridLayout(self.subtalker_body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(m.sp(0.6))
        grid.setVerticalSpacing(m.sp(0.35))

        self.subtalker_temperature = QDoubleSpinBox()
        self.subtalker_temperature.setRange(0.0, 2.0)
        self.subtalker_temperature.setSingleStep(0.05)
        self.subtalker_temperature.setDecimals(2)

        self.subtalker_top_p = QDoubleSpinBox()
        self.subtalker_top_p.setRange(0.05, 1.0)
        self.subtalker_top_p.setSingleStep(0.05)
        self.subtalker_top_p.setDecimals(2)

        self.subtalker_top_k = QSpinBox()
        self.subtalker_top_k.setRange(0, 200)

        self.subtalker_do_sample = QCheckBox("Sample")
        self.subtalker_do_sample.setToolTip(
            "Off is greedy: the detail stage always takes its most likely code."
        )

        for i, (label, widget) in enumerate([
            ("Sub temperature", self.subtalker_temperature),
            ("Sub top-p", self.subtalker_top_p),
            ("Sub top-k", self.subtalker_top_k),
        ]):
            row, col = divmod(i, 2)
            cell = QVBoxLayout()
            cell.setSpacing(metrics().sp(0.12))
            cell.addWidget(_hint(label))
            cell.addWidget(widget)
            holder = QWidget()
            holder.setLayout(cell)
            grid.addWidget(holder, row, col)
        grid.addWidget(self.subtalker_do_sample, 1, 1)

        self.subtalker_body.setVisible(False)
        box.addWidget(self.subtalker_body)
        self._on_link_toggled(True)
        return container

    def _on_subtalker_toggled(self, expanded: bool) -> None:
        self.subtalker_toggle.setText(("▾ " if expanded else "▸ ") + "Sub-talker")
        self.subtalker_body.setVisible(expanded)

    def _on_link_toggled(self, linked: bool) -> None:
        """Linked mirrors the main controls, which is the model's own default."""
        for widget in (
            self.subtalker_temperature,
            self.subtalker_top_p,
            self.subtalker_top_k,
            self.subtalker_do_sample,
        ):
            widget.setEnabled(not linked)
        if linked:
            self._mirror_main_to_subtalker()

    def _mirror_main_to_subtalker(self) -> None:
        self.subtalker_temperature.setValue(self.temperature.value())
        self.subtalker_top_p.setValue(self.top_p.value())
        self.subtalker_top_k.setValue(self.top_k.value())
        self.subtalker_do_sample.setChecked(self.temperature.value() > 0)

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
        if self.subtalker_link.isChecked():
            self._mirror_main_to_subtalker()
        return {
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "top_k": self.top_k.value(),
            "repetition_penalty": self.repetition_penalty.value(),
            "max_new_tokens": self.max_new_tokens.value(),
            "subtalker_do_sample": self.subtalker_do_sample.isChecked(),
            "subtalker_temperature": self.subtalker_temperature.value(),
            "subtalker_top_p": self.subtalker_top_p.value(),
            "subtalker_top_k": self.subtalker_top_k.value(),
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

        # Only treat sub-talker values as independent if they were actually stored
        # that way; otherwise leave them mirroring the main controls.
        if any(k in values for k in SUBTALKER_KEYS):
            self.subtalker_temperature.setValue(
                float(values.get("subtalker_temperature", DEFAULTS["subtalker_temperature"]))
            )
            self.subtalker_top_p.setValue(
                float(values.get("subtalker_top_p", DEFAULTS["subtalker_top_p"]))
            )
            self.subtalker_top_k.setValue(
                int(values.get("subtalker_top_k", DEFAULTS["subtalker_top_k"]))
            )
            self.subtalker_do_sample.setChecked(
                bool(values.get("subtalker_do_sample", DEFAULTS["subtalker_do_sample"]))
            )
        elif self.subtalker_link.isChecked():
            self._mirror_main_to_subtalker()

    def set_linked(self, linked: bool) -> None:
        self.subtalker_link.setChecked(linked)

    def reset_defaults(self) -> None:
        self.subtalker_link.setChecked(True)
        self.apply(DEFAULTS)
