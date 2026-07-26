"""Font-relative sizing so the UI scales with the user's Qt/HiDPI settings.

Every size in the app derives from the application font rather than a hardcoded
pixel count. Qt scales device-independent pixels by `devicePixelRatio`, but it
does *not* scale them by the user's font DPI (`QT_FONT_DPI`) or their chosen
system font size — which is how most HiDPI desktops actually scale text. Sizing
off the font metrics covers all three cases at once.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import QApplication

_current: "Metrics | None" = None


class Metrics:
    def __init__(self, font: QFont):
        fm = QFontMetrics(font)
        pt = font.pointSizeF()
        # Fall back through pointSize -> pixelSize -> a sane default.
        if pt <= 0:
            pt = font.pixelSize() * 0.75 if font.pixelSize() > 0 else 10.0
        self.base_pt = pt
        self.line = max(12, fm.height())
        self.char = max(5, fm.horizontalAdvance("0"))

    def pt(self, ratio: float = 1.0) -> float:
        """A font size in points, relative to the application font."""
        return round(self.base_pt * ratio, 1)

    def sp(self, ratio: float = 1.0) -> int:
        """Spacing/padding derived from the line height."""
        return max(1, round(self.line * ratio))

    def ch(self, count: float) -> int:
        """A width that fits roughly `count` digits, plus padding."""
        return round(self.char * count) + self.sp(0.6)

    def icon_button(self) -> int:
        """Square-ish button that always fits a single glyph."""
        return max(self.sp(1.7), self.ch(2))


def metrics() -> Metrics:
    global _current
    if _current is None:
        app = QApplication.instance()
        _current = Metrics(app.font() if app is not None else QFont())
    return _current


def refresh() -> Metrics:
    """Recompute after the application font changes."""
    global _current
    _current = None
    return metrics()
