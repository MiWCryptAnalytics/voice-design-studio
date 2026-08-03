"""Small shared widget subclasses that keep Qt's helper popups on-theme."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QStyledItemDelegate, QWidget


class ThemedComboBox(QComboBox):
    """A combo box whose popup list follows the application font.

    QComboBox's default popup paints items through the *menu* delegate, whose
    font and row metrics come from the platform menu theme captured at
    startup — not from the (possibly auto-scaled) application font, and not
    from the popup view's own font. At a 2x UI scale that leaves popup text
    at half the size of everything else. Two changes fix it deterministically:
    a styled view delegate, which paints and measures with the view's font,
    and syncing that font from the combo just before the popup opens.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setItemDelegate(QStyledItemDelegate(self))

    def showPopup(self) -> None:  # noqa: N802 - Qt naming
        view = self.view()
        if view is not None and view.font() != self.font():
            view.setFont(self.font())
        super().showPopup()
