"""Cast: map the speakers found in a dialogue script to saved voices.

Appears only when the script actually uses speaker labels. Consistency across a
character's lines comes from reusing that character's description, so a cast
entry is just a preset reference.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.library import VoiceLibrary
from ..core.templates import load_templates
from . import theme
from .metrics import metrics
from .widgets import ThemedComboBox

UNASSIGNED = "— choose a voice —"
CURRENT_DESIGN = "Current design panel voice"


class CastPanel(QFrame):
    castChanged = pyqtSignal()
    auditionRequested = pyqtSignal(str, str)  # instruct, display name

    def __init__(self, library: VoiceLibrary, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.library = library
        self.templates = load_templates()
        self._cast: dict[str, str] = {}
        self._speakers: list[str] = []
        self._loading = False

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.15))

        header = QHBoxLayout()
        title = QLabel("CAST")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.status = QLabel("")
        self.status.setProperty("role", "hint")
        header.addWidget(self.status)
        layout.addLayout(header)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Speaker", "Voice", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(m.sp(4.5))
        layout.addWidget(self.table, 1)

        self.lock_voices = QCheckBox("Keep each voice steady across its lines")
        self.lock_voices.setChecked(True)
        self.lock_voices.setToolTip(
            "Renders all of a speaker's lines in one pass and splits them apart.\n"
            "A voice only stays genuinely fixed inside a single generation — this\n"
            "checkpoint has no speaker-locking mechanism — so this is what stops a\n"
            "character drifting between lines.\n\n"
            "If a split can't be made safely that group falls back to line-by-line."
        )
        layout.addWidget(self.lock_voices)

    # ---------- data ----------

    @property
    def cast(self) -> dict[str, str]:
        return dict(self._cast)

    def set_cast(self, cast: dict[str, str]) -> None:
        self._cast = dict(cast or {})
        if self._speakers:
            self.set_speakers(self._speakers)

    def voice_options(self) -> list[tuple[str | None, str, bool]]:
        """(key, display, selectable) — your saved voices first, then templates."""
        options: list[tuple[str | None, str, bool]] = [
            ("", UNASSIGNED, True),
            ("__current__", CURRENT_DESIGN, True),
        ]
        presets = self.library.all()
        if presets:
            options.append((None, "— My voices —", False))
            for preset in presets:
                star = "★ " if preset.favorite else ""
                pin = "  📌" if preset.seed_pinned else ""
                options.append((f"preset:{preset.id}", f"{star}{preset.name}{pin}", True))
        options.append((None, "— Templates —", False))
        for template in self.templates.templates:
            options.append((f"template:{template.id}", template.name, True))
        return options

    def resolves(self, key: str) -> bool:
        """False once the preset behind an assignment has been deleted."""
        if not key or key == "__current__":
            return bool(key)
        kind, _, ident = key.partition(":")
        if kind == "preset":
            return self.library.get(ident) is not None
        if kind == "template":
            return self.templates.get(ident) is not None
        return False

    def _fill_combo(self, combo: QComboBox, selected: str) -> None:
        combo.clear()
        for key, display, selectable in self.voice_options():
            combo.addItem(display, key)
            if not selectable:
                index = combo.count() - 1
                combo.setItemData(index, 0, Qt.ItemDataRole.UserRole - 1)
                font = combo.font()
                font.setBold(True)
                combo.setItemData(index, font, Qt.ItemDataRole.FontRole)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))

    def refresh_voices(self) -> None:
        """Rebuild the voice lists after the library changes.

        Selections are preserved, except where the preset behind one has been
        deleted — those revert to unassigned so generation refuses, rather than
        silently substituting a different voice.
        """
        dropped = [s for s, key in self._cast.items() if not self.resolves(key)]
        for speaker in dropped:
            self._cast.pop(speaker, None)

        self._loading = True
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            speaker = self.table.item(row, 0).text()
            if isinstance(combo, QComboBox):
                self._fill_combo(combo, self._cast.get(speaker, ""))
        self._loading = False

        self._update_status()
        if dropped:
            self.status.setText(
                f"Voice deleted — reassign: {', '.join(dropped[:3])}"
            )
            self.status.setStyleSheet(f"color: {theme.BAD};")
            self.castChanged.emit()

    def instruct_for(self, key: str, fallback: str = "") -> str:
        if not key or key == "__current__":
            return fallback
        kind, _, ident = key.partition(":")
        if kind == "preset":
            preset = self.library.get(ident)
            return preset.instruct if preset else fallback
        if kind == "template":
            template = self.templates.get(ident)
            return template.instruct if template else fallback
        return fallback

    def set_speakers(self, speakers: list[str]) -> None:
        self._loading = True
        self._speakers = list(speakers)

        self.table.setRowCount(len(speakers))
        for row, speaker in enumerate(speakers):
            name = QTableWidgetItem(speaker)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, name)

            combo = ThemedComboBox()
            self._fill_combo(combo, self._cast.get(speaker, ""))
            combo.currentIndexChanged.connect(
                lambda _idx, s=speaker, c=combo: self._on_assigned(s, c)
            )
            self.table.setCellWidget(row, 1, combo)

            audition = QPushButton("▶")
            audition.setProperty("role", "icon")
            audition.setToolTip(f"Audition the voice assigned to {speaker}")
            audition.clicked.connect(lambda _c, s=speaker: self._audition(s))
            self.table.setCellWidget(row, 2, audition)

        self._size_rows()
        self._loading = False
        self._update_status()

    def _size_rows(self) -> None:
        """Make rows tall enough for their styled cell widgets.

        The table computes default row heights from the bare style, but the
        combos and buttons in the cells are sized by the QSS, whose padding
        grows with the application font. At HiDPI scale factors the widgets end
        up taller than the rows and spill over each other unless the rows are
        sized from the widgets that actually live in them.
        """
        rows = self.table.rowCount()
        if not rows:
            return

        need = 0
        for row in range(rows):
            for col in (1, 2):
                widget = self.table.cellWidget(row, col)
                if widget is not None:
                    need = max(need, widget.sizeHint().height(),
                               widget.minimumSizeHint().height())
        row_height = need + metrics().sp(0.15)
        self.table.verticalHeader().setDefaultSectionSize(row_height)
        for row in range(rows):
            self.table.setRowHeight(row, row_height)

        # Show every speaker up to four without a scrollbar, then scroll.
        frame = 2 * self.table.frameWidth()
        header_height = self.table.horizontalHeader().sizeHint().height()
        shown = min(rows, 4)
        target = header_height + row_height * shown + frame
        self.table.setMinimumHeight(
            min(target, header_height + row_height * 2 + frame)
        )
        self.table.setMaximumHeight(target)

    def _on_assigned(self, speaker: str, combo: QComboBox) -> None:
        if self._loading:
            return
        key = combo.currentData() or ""
        if key:
            self._cast[speaker] = key
        else:
            self._cast.pop(speaker, None)
        self._update_status()
        self.castChanged.emit()

    def _audition(self, speaker: str) -> None:
        instruct = self.instruct_for(self._cast.get(speaker, ""))
        if not instruct:
            self.status.setText(f"Assign a voice to {speaker} first")
            return
        self.auditionRequested.emit(instruct, speaker)

    def unassigned(self) -> list[str]:
        return [s for s in self._speakers if not self._cast.get(s)]

    def _update_status(self) -> None:
        missing = self.unassigned()
        if not self._speakers:
            self.status.setText("no speakers detected")
            self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        elif missing:
            self.status.setText(f"{len(missing)} unassigned: {', '.join(missing[:3])}")
            self.status.setStyleSheet(f"color: {theme.WARN};")
        else:
            self.status.setText(f"{len(self._speakers)} voices assigned")
            self.status.setStyleSheet(f"color: {theme.GOOD};")
