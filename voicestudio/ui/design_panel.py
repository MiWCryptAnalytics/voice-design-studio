"""Left column: template picker, trait builder, instruct editor, voice library."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import traits as traits_mod
from ..core.library import VoiceLibrary, VoicePreset
from ..core.templates import VoiceTemplate, load_templates
from . import theme
from .metrics import metrics


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


class SavePresetDialog(QDialog):
    """Name a voice and decide whether to pin the seed that produced it."""

    def __init__(self, suggested_name: str, seed: int | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save voice")
        self._seed = seed

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        layout.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit(suggested_name)
        self.name_edit.selectAll()
        self.name_edit.setMinimumWidth(m.ch(34))
        layout.addWidget(self.name_edit)

        self.pin_check = QCheckBox(
            f"Pin seed {seed}" if seed is not None else "Pin seed (none available)"
        )
        self.pin_check.setEnabled(seed is not None)
        self.pin_check.setChecked(seed is not None)
        layout.addWidget(self.pin_check)

        explain = QLabel(
            "The description is what carries the voice — it stays recognisable "
            "at any seed.\nPinning additionally reproduces this exact rendition "
            "when the text is unchanged."
        )
        explain.setProperty("role", "hint")
        explain.setWordWrap(True)
        layout.addWidget(explain)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def preset_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def pinned_seed(self) -> int | None:
        return self._seed if self.pin_check.isChecked() else None


class DesignPanel(QFrame):
    instructChanged = pyqtSignal(str)
    auditionRequested = pyqtSignal(str, str)  # instruct, display name
    seedApplied = pyqtSignal(int)  # a loaded preset pinned this seed

    def __init__(self, library: VoiceLibrary, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.library = library
        self.templates = load_templates()
        # Set by MainWindow — the seed that produced what you last heard.
        self.seed_provider: Callable[[], int] | None = None

        self._detached = False  # instruct text edited by hand
        self._syncing = False  # guard against feedback loops

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.6))

        layout.addWidget(_title("VOICE DESIGN"))
        layout.addLayout(self._build_template_row())
        layout.addWidget(self._divider())
        layout.addWidget(_title("TRAITS"))
        layout.addLayout(self._build_traits_grid())
        layout.addLayout(self._build_trait_actions())
        layout.addWidget(self._divider())

        header = QHBoxLayout()
        header.addWidget(_title("VOICE DESCRIPTION"))
        header.addStretch(1)
        self.detach_hint = QLabel("")
        self.detach_hint.setProperty("role", "hint")
        header.addWidget(self.detach_hint)
        layout.addLayout(header)

        self.instruct_edit = QPlainTextEdit()
        self.instruct_edit.setPlaceholderText(
            "Describe the voice… e.g. “A warm, gravelly older man speaking slowly.”\n"
            "Pick a template above to start, then edit freely."
        )
        self.instruct_edit.setMinimumHeight(metrics().sp(5.0))
        self.instruct_edit.textChanged.connect(self._on_instruct_edited)
        layout.addWidget(self.instruct_edit)

        layout.addWidget(self._divider())
        layout.addLayout(self._build_library_section(), 1)

    # ---------- construction helpers ----------

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {theme.BORDER}; border: none;")
        return line

    def _build_template_row(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(metrics().sp(0.35))

        self.template_search = QLineEdit()
        self.template_search.setPlaceholderText("Search templates…")
        self.template_search.textChanged.connect(self._populate_templates)

        self.template_combo = QComboBox()
        self.template_combo.activated.connect(self._on_template_chosen)

        self.audition_button = QPushButton("Audition")
        self.audition_button.setToolTip(
            "Generate this template against a fixed demo line and seed"
        )
        self.audition_button.clicked.connect(self._on_audition)

        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.35))
        row.addWidget(self.template_combo, 1)
        row.addWidget(self.audition_button)

        box.addWidget(self.template_search)
        box.addLayout(row)
        self._populate_templates()
        return box

    def _populate_templates(self) -> None:
        query = self.template_search.text()
        grouped = self.templates.by_category(query)

        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self.template_combo.addItem("Choose a template…", None)

        header_font = QFont()
        header_font.setBold(True)
        count = 0
        for category in self.templates.categories():
            items = grouped.get(category, [])
            if not items:
                continue
            self.template_combo.addItem(f"— {category} —", None)
            index = self.template_combo.count() - 1
            self.template_combo.setItemData(
                index, 0, Qt.ItemDataRole.UserRole - 1  # non-selectable
            )
            self.template_combo.setItemData(
                index, header_font, Qt.ItemDataRole.FontRole
            )
            for t in items:
                self.template_combo.addItem(f"   {t.name}", t.id)
                self.template_combo.setItemData(
                    self.template_combo.count() - 1,
                    t.instruct,
                    Qt.ItemDataRole.ToolTipRole,
                )
                count += 1

        if count == 0:
            self.template_combo.addItem("No matches", None)
        self.template_combo.blockSignals(False)

    def _build_traits_grid(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(metrics().sp(0.45))
        grid.setVerticalSpacing(metrics().sp(0.35))
        self.trait_combos: dict[str, QComboBox] = {}

        for i, dim in enumerate(traits_mod.DIMENSIONS):
            row, col = divmod(i, 2)
            label = QLabel(dim.label)
            label.setProperty("role", "hint")
            combo = QComboBox()
            combo.addItems(list(dim.options))
            combo.currentIndexChanged.connect(self._on_trait_changed)
            self.trait_combos[dim.key] = combo

            cell = QVBoxLayout()
            cell.setSpacing(metrics().sp(0.12))
            cell.addWidget(label)
            cell.addWidget(combo)
            container = QWidget()
            container.setLayout(cell)
            grid.addWidget(container, row, col)

        return grid

    def _build_trait_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.35))

        self.recompose_button = QPushButton("Recompose from traits")
        self.recompose_button.setToolTip(
            "Overwrite the description with a sentence built from the traits above"
        )
        self.recompose_button.clicked.connect(self._recompose)

        clear = QPushButton("Clear traits")
        clear.clicked.connect(self._clear_traits)

        row.addWidget(self.recompose_button, 1)
        row.addWidget(clear)
        return row

    def _build_library_section(self) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setSpacing(metrics().sp(0.35))

        header = QHBoxLayout()
        header.addWidget(_title("MY VOICES"))
        header.addStretch(1)
        self.library_count = QLabel("")
        self.library_count.setProperty("role", "hint")
        header.addWidget(self.library_count)
        box.addLayout(header)

        self.library_search = QLineEdit()
        self.library_search.setPlaceholderText("Search saved voices…")
        self.library_search.textChanged.connect(self.refresh_library)
        box.addWidget(self.library_search)

        self.library_list = QListWidget()
        self.library_list.setMinimumHeight(metrics().sp(5.0))
        self.library_list.itemDoubleClicked.connect(self._load_selected_preset)
        box.addWidget(self.library_list, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(metrics().sp(0.35))

        save = QPushButton("Save as…")
        save.setToolTip("Save the current description as a preset (Ctrl+S)")
        save.clicked.connect(self.save_current_as_preset)

        load = QPushButton("Load")
        load.clicked.connect(self._load_selected_preset)

        star = QPushButton("★")
        star.setProperty("role", "icon")
        star.setToolTip("Toggle favorite")
        star.clicked.connect(self._toggle_favorite)

        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.clicked.connect(self._delete_selected)

        for widget in (save, load, star, delete):
            buttons.addWidget(widget)
        box.addLayout(buttons)

        self.refresh_library()
        return box

    # ---------- state ----------

    @property
    def instruct(self) -> str:
        return self.instruct_edit.toPlainText().strip()

    def set_instruct(self, text: str, *, detached: bool = True) -> None:
        self._syncing = True
        self.instruct_edit.setPlainText(text)
        self._syncing = False
        self._detached = detached
        self._update_detach_hint()
        self.instructChanged.emit(self.instruct)

    def current_traits(self) -> dict[str, str]:
        return {key: combo.currentText() for key, combo in self.trait_combos.items()}

    def _apply_traits(self, values: dict) -> None:
        normalized = traits_mod.normalize(values)
        for key, combo in self.trait_combos.items():
            combo.blockSignals(True)
            combo.setCurrentText(normalized.get(key, traits_mod.ANY))
            combo.blockSignals(False)

    # ---------- events ----------

    def _on_template_chosen(self, index: int) -> None:
        template_id = self.template_combo.itemData(index)
        if not template_id:
            return
        template = self.templates.get(template_id)
        if template is None:
            return
        self._apply_traits(template.traits)
        # A template is a starting point: text and traits agree, so not detached.
        self.set_instruct(template.instruct, detached=False)

    def _on_audition(self) -> None:
        template_id = self.template_combo.currentData()
        template: VoiceTemplate | None = (
            self.templates.get(template_id) if template_id else None
        )
        instruct = template.instruct if template else self.instruct
        name = template.name if template else "Current description"
        if not instruct:
            QMessageBox.information(
                self, "Nothing to audition", "Pick a template or write a description."
            )
            return
        self.auditionRequested.emit(instruct, name)

    def _on_trait_changed(self) -> None:
        if self._detached:
            # Don't clobber hand-written text; nudge instead.
            self._update_detach_hint()
            return
        self._recompose(keep_attached=True)

    def _recompose(self, keep_attached: bool = True) -> None:
        text = traits_mod.compose(self.current_traits())
        if not text:
            return
        self.set_instruct(text, detached=not keep_attached)

    def _clear_traits(self) -> None:
        self._apply_traits(traits_mod.empty_traits())
        self._update_detach_hint()

    def _on_instruct_edited(self) -> None:
        if self._syncing:
            return
        self._detached = True
        self._update_detach_hint()
        self.instructChanged.emit(self.instruct)

    def _update_detach_hint(self) -> None:
        if self._detached:
            self.detach_hint.setText("edited by hand")
            self.detach_hint.setStyleSheet(f"color: {theme.WARN};")
            self.recompose_button.setStyleSheet(
                f"border: 1px solid {theme.WARN}; color: {theme.WARN};"
            )
        else:
            self.detach_hint.setText("following traits")
            self.detach_hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self.recompose_button.setStyleSheet("")

    # ---------- library ----------

    def refresh_library(self) -> None:
        self.library_list.clear()
        presets = self.library.all(self.library_search.text())
        for preset in presets:
            prefix = "★ " if preset.favorite else "   "
            # The pin marks presets that reproduce one exact voice.
            suffix = f"  📌 {preset.seed}" if preset.seed_pinned else "  (any seed)"
            item = QListWidgetItem(f"{prefix}{preset.name}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, preset.id)
            item.setToolTip(
                f"{preset.instruct}\n\n"
                + (
                    f"Pinned to seed {preset.seed} — reproduces this exact voice."
                    if preset.seed_pinned
                    else "No seed pinned — each load is a new speaker matching the description."
                )
            )
            if preset.favorite:
                item.setForeground(Qt.GlobalColor.yellow)
            self.library_list.addItem(item)
        total = len(self.library.all())
        self.library_count.setText(f"{total} saved" if total else "none yet")

    def _selected_preset_id(self) -> str | None:
        item = self.library_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def save_current_as_preset(self) -> None:
        instruct = self.instruct
        if not instruct:
            QMessageBox.information(
                self, "Nothing to save", "Write or pick a voice description first."
            )
            return

        template_id = self.template_combo.currentData()
        template = self.templates.get(template_id) if template_id else None
        suggested = template.name if template else "My voice"

        seed = self.seed_provider() if self.seed_provider else None
        dialog = SavePresetDialog(suggested, seed, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.preset_name:
            return

        preset = VoicePreset(
            name=dialog.preset_name,
            instruct=instruct,
            traits=self.current_traits(),
            source_template=template_id,
            seed=dialog.pinned_seed,
        )
        self.library.add(preset)
        self.refresh_library()

    def _load_selected_preset(self) -> None:
        preset_id = self._selected_preset_id()
        if preset_id:
            self.load_preset(preset_id)

    def load_preset(self, preset_id: str) -> bool:
        preset = self.library.get(preset_id)
        if preset is None:
            return False
        self._apply_traits(preset.traits)
        self.set_instruct(preset.instruct, detached=True)
        if preset.seed_pinned:
            self.seedApplied.emit(preset.seed)
        return True

    def _toggle_favorite(self) -> None:
        preset_id = self._selected_preset_id()
        if preset_id:
            self.library.toggle_favorite(preset_id)
            self.refresh_library()

    def _delete_selected(self) -> None:
        preset_id = self._selected_preset_id()
        preset = self.library.get(preset_id) if preset_id else None
        if preset is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete voice",
            f"Delete “{preset.name}”? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.library.remove(preset.id)
            self.refresh_library()
