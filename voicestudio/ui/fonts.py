"""Load the bundled UI typeface: IBM Plex Sans (SIL OFL 1.1).

Bundling the typeface makes the UI render identically on every machine
instead of taking whatever sans-serif the desktop resolves. Only the family
changes — the desktop's chosen point size is kept, and HiDPI auto-scaling
(scaling.py) multiplies it afterwards. Regular, SemiBold and Bold faces cover
everything the stylesheet and item fonts ask for (weights 400/600/700).
"""

from __future__ import annotations

from PyQt6.QtGui import QFontDatabase

from ..core.config import ASSETS_DIR

FONTS_DIR = ASSETS_DIR / "fonts"
FAMILY = "IBM Plex Sans"


def install_application_font(app) -> str | None:
    """Register the bundled faces and switch the application font family.

    Returns the family name on success, or None when the bundle is missing or
    unloadable — the system font stays, a cosmetic downgrade rather than an
    error. Call right after the QApplication exists, before auto-scaling and
    before the stylesheet is built (both read the application font).
    """
    loaded: set[str] = set()
    for path in sorted(FONTS_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            loaded.update(QFontDatabase.applicationFontFamilies(font_id))
    if FAMILY not in loaded:
        return None

    font = app.font()
    font.setFamily(FAMILY)
    app.setFont(font)
    return FAMILY
