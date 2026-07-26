"""Right column: take history, A/B comparison, per-take actions."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.history import Take, TakeHistory
from . import theme
from .metrics import metrics


class TakeCard(QFrame):
    """One generation: what it was, and what you can do with it."""

    playRequested = pyqtSignal(str)
    starToggled = pyqtSignal(str)
    slotAssigned = pyqtSignal(str, str)  # take id, "A" | "B"
    rerollRequested = pyqtSignal(str)
    restoreRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, take: Take, slot: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.take = take
        self.setProperty("role", "card")
        if slot:
            self.setProperty("slot", slot)

        m = metrics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(0.6), m.sp(0.45), m.sp(0.6), m.sp(0.45))
        layout.setSpacing(m.sp(0.35))

        top = QHBoxLayout()
        top.setSpacing(m.sp(0.35))

        title = QLabel(take.title)
        title.setWordWrap(True)
        title.setToolTip(take.instruct or "(no description)")
        top.addWidget(title, 1)

        self.star_button = QPushButton("★" if take.starred else "☆")
        self.star_button.setProperty("role", "icon")
        self.star_button.setMinimumWidth(m.icon_button())
        self.star_button.setToolTip("Keep this take")
        if take.starred:
            self.star_button.setStyleSheet(f"color: {theme.STAR};")
        self.star_button.clicked.connect(lambda: self.starToggled.emit(take.id))
        top.addWidget(self.star_button)
        layout.addLayout(top)

        spoken = " ".join(take.text.split())
        if spoken:
            excerpt = QLabel(f"“{spoken[:70]}{'…' if len(spoken) > 70 else ''}”")
            excerpt.setProperty("role", "hint")
            excerpt.setWordWrap(True)
            layout.addWidget(excerpt)

        meta = QLabel(
            f"{take.when} · {take.duration:.1f}s · seed {take.seed} · "
            f"T{take.temperature:g} · {take.language}"
        )
        meta.setProperty("role", "metric")
        layout.addWidget(meta)

        layout.addLayout(self._build_actions())

    def _build_actions(self) -> QHBoxLayout:
        m = metrics()
        row = QHBoxLayout()
        row.setSpacing(m.sp(0.22))
        size = m.icon_button()

        def button(text: str, tip: str, handler) -> QPushButton:
            btn = QPushButton(text)
            btn.setProperty("role", "icon")
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            btn.setMinimumWidth(size)
            return btn

        take_id = self.take.id
        row.addWidget(button("▶", "Play", lambda: self.playRequested.emit(take_id)))
        row.addWidget(button("A", "Pin as A", lambda: self.slotAssigned.emit(take_id, "A")))
        row.addWidget(button("B", "Pin as B", lambda: self.slotAssigned.emit(take_id, "B")))
        row.addWidget(button("⟳", "Re-roll: same settings, new seed",
                             lambda: self.rerollRequested.emit(take_id)))
        row.addWidget(button("↩", "Restore these settings into the editor",
                             lambda: self.restoreRequested.emit(take_id)))
        row.addStretch(1)
        row.addWidget(button("⤓", "Export WAV…", lambda: self.exportRequested.emit(take_id)))
        row.addWidget(button("✕", "Delete", lambda: self.deleteRequested.emit(take_id)))
        return row


class TakesPanel(QFrame):
    playRequested = pyqtSignal(str)
    rerollRequested = pyqtSignal(str)
    restoreRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)

    def __init__(self, history: TakeHistory, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.history = history
        self.slot_a: str | None = None
        self.slot_b: str | None = None

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        title = QLabel("TAKES")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setProperty("role", "hint")
        header.addWidget(self.count_label)
        clear = QPushButton("Clear unstarred")
        clear.setProperty("role", "danger")
        clear.clicked.connect(self._clear_unstarred)
        header.addWidget(clear)
        layout.addLayout(header)

        layout.addWidget(self._build_compare_bar())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.cards_layout = QVBoxLayout(self.container)
        self.cards_layout.setContentsMargins(0, 0, m.sp(0.35), 0)
        self.cards_layout.setSpacing(m.sp(0.35))
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

        self.refresh()

    def _build_compare_bar(self) -> QFrame:
        m = metrics()
        bar = QFrame()
        bar.setProperty("role", "card")
        row = QHBoxLayout(bar)
        row.setContentsMargins(m.sp(0.45), m.sp(0.35), m.sp(0.45), m.sp(0.35))
        row.setSpacing(m.sp(0.35))
        edge = m.sp(0.18)

        self.a_button = QPushButton("A —")
        self.a_button.setToolTip("Play take pinned as A (Ctrl+1)")
        self.a_button.setStyleSheet(f"border-left: {edge}px solid {theme.SLOT_A};")
        self.a_button.clicked.connect(lambda: self._play_slot("A"))

        self.b_button = QPushButton("B —")
        self.b_button.setToolTip("Play take pinned as B (Ctrl+2)")
        self.b_button.setStyleSheet(f"border-left: {edge}px solid {theme.SLOT_B};")
        self.b_button.clicked.connect(lambda: self._play_slot("B"))

        row.addWidget(self.a_button, 1)
        row.addWidget(self.b_button, 1)
        return bar

    # ---------- data ----------

    def refresh(self) -> None:
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        takes = self.history.all()
        for take in takes:
            slot = "A" if take.id == self.slot_a else ("B" if take.id == self.slot_b else "")
            card = TakeCard(take, slot)
            card.playRequested.connect(self.playRequested.emit)
            card.rerollRequested.connect(self.rerollRequested.emit)
            card.restoreRequested.connect(self.restoreRequested.emit)
            card.exportRequested.connect(self.exportRequested.emit)
            card.starToggled.connect(self._toggle_star)
            card.slotAssigned.connect(self._assign_slot)
            card.deleteRequested.connect(self._delete)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

        starred = sum(1 for t in takes if t.starred)
        self.count_label.setText(
            f"{len(takes)} take{'s' if len(takes) != 1 else ''}"
            + (f" · {starred} kept" if starred else "")
        )
        self._update_compare_bar()

    def _update_compare_bar(self) -> None:
        for slot, button in (("A", self.a_button), ("B", self.b_button)):
            take_id = self.slot_a if slot == "A" else self.slot_b
            take = self.history.get(take_id) if take_id else None
            if take is None:
                button.setText(f"{slot} —")
                button.setEnabled(False)
            else:
                label = take.title
                button.setText(f"{slot}  {label[:28]}{'…' if len(label) > 28 else ''}")
                button.setToolTip(f"{take.instruct}\nseed {take.seed}")
                button.setEnabled(True)

    def _assign_slot(self, take_id: str, slot: str) -> None:
        if slot == "A":
            # Don't let one take occupy both slots — that defeats comparison.
            if self.slot_b == take_id:
                self.slot_b = None
            self.slot_a = take_id
        else:
            if self.slot_a == take_id:
                self.slot_a = None
            self.slot_b = take_id
        self.refresh()

    def play_slot(self, slot: str) -> None:
        self._play_slot(slot)

    def _play_slot(self, slot: str) -> None:
        take_id = self.slot_a if slot == "A" else self.slot_b
        if take_id and self.history.get(take_id):
            self.playRequested.emit(take_id)

    def _toggle_star(self, take_id: str) -> None:
        self.history.toggle_star(take_id)
        self.refresh()

    def _delete(self, take_id: str) -> None:
        if self.slot_a == take_id:
            self.slot_a = None
        if self.slot_b == take_id:
            self.slot_b = None
        self.history.remove(take_id)
        self.refresh()

    def _clear_unstarred(self) -> None:
        self.history.clear_unstarred()
        remaining = {t.id for t in self.history.all()}
        if self.slot_a not in remaining:
            self.slot_a = None
        if self.slot_b not in remaining:
            self.slot_b = None
        self.refresh()
