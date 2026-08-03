"""Entry point: python -m voicestudio"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication

from . import APP_NAME


def main() -> int:
    # Must be set before the QApplication exists. PassThrough keeps fractional
    # scale factors (1.25, 1.5, …) intact instead of rounding them to integers,
    # which is what makes the UI look either cramped or bloated on HiDPI.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # Imported after the QApplication so sizing reads the real application font.
    from .ui import MainWindow
    from .ui.fonts import install_application_font
    from .ui.metrics import refresh
    from .ui.scaling import apply_auto_scale
    from .ui.theme import build_qss

    # Order matters: the bundled typeface first (family change only), then the
    # HiDPI auto-scale (multiplies the size), then metrics/QSS which bake both
    # into every derived dimension.
    install_application_font(app)
    apply_auto_scale(app)

    app.setStyleSheet(build_qss(refresh()))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
