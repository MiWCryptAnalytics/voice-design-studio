"""Verify the UI scales with the application font instead of fixed pixels.

Builds the panels at several font sizes and checks that text still fits, that
fonts actually grow, and that spacing scales with them.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/hidpi_check.py
"""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.history import TakeHistory  # noqa: E402
from voicestudio.core.library import VoiceLibrary  # noqa: E402
from voicestudio.ui.design_panel import DesignPanel  # noqa: E402
from voicestudio.ui.metrics import refresh  # noqa: E402
from voicestudio.ui.params_panel import ParamsPanel  # noqa: E402
from voicestudio.ui.player import PlayerBar  # noqa: E402
from voicestudio.ui.script_panel import ScriptPanel  # noqa: E402
from voicestudio.ui.takes_panel import TakesPanel  # noqa: E402
from voicestudio.ui.theme import build_qss  # noqa: E402

FONT_SIZES = [9.0, 12.0, 16.0, 22.0]
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def build_ui(app) -> tuple[QWidget, dict]:
    """Panels only — MainWindow would load the model."""
    library, history = VoiceLibrary(), TakeHistory()
    design = DesignPanel(library)
    script = ScriptPanel()
    params = ParamsPanel()
    player = PlayerBar()
    takes = TakesPanel(history)

    root = QWidget()
    row = QHBoxLayout(root)
    center = QVBoxLayout()
    center.addWidget(script)
    center.addWidget(params)
    center.addWidget(player)
    row.addWidget(design, 3)
    row.addLayout(center, 4)
    row.addWidget(takes, 3)
    return root, {"design": design, "script": script, "params": params, "takes": takes}


def clipped(root: QWidget) -> list[str]:
    """Controls whose preferred width exceeds the width they actually got."""
    bad = []
    for widget in root.findChildren((QPushButton, QComboBox)):
        if not widget.isVisible():
            continue
        want = widget.sizeHint().width()
        got = widget.width()
        if want > got + 1:
            text = widget.text() if isinstance(widget, QPushButton) else widget.currentText()
            bad.append(f"{text!r} wants {want}px, got {got}px")
    return bad


def main() -> int:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    check(
        "rounding policy is PassThrough (fractional scaling preserved)",
        QGuiApplication.highDpiScaleFactorRoundingPolicy()
        == Qt.HighDpiScaleFactorRoundingPolicy.PassThrough,
    )

    observed = {}
    for pt in FONT_SIZES:
        print(f"\n@ {pt:g}pt application font")
        font = QFont(app.font())
        font.setPointSizeF(pt)
        app.setFont(font)
        m = refresh()
        app.setStyleSheet(build_qss(m))

        root, panels = build_ui(app)
        # Scale the container the way the real window does.
        root.resize(m.ch(185), m.sp(52))
        root.show()
        app.processEvents()

        # Unstyled inputs must inherit the application font, not a pinned size.
        probe = panels["script"].text_edit
        check(
            "editor inherits application font",
            abs(probe.font().pointSizeF() - pt) < 0.51,
            f"{probe.font().pointSizeF():g}pt vs {pt:g}pt",
        )

        # Role-styled labels scale relative to it.
        title = next(
            lab for lab in root.findChildren(QLabel)
            if lab.property("role") == "title"
        )
        title_pt = title.fontInfo().pointSizeF()
        observed[pt] = (title_pt, m.line, m.sp(0.7))
        check("title label scales with font", title_pt >= pt * 0.75,
              f"{title_pt:g}pt")

        bad = clipped(root)
        check("no clipped buttons or combos", not bad,
              "; ".join(bad[:3]) if bad else "")

        root.close()
        root.deleteLater()
        app.processEvents()

    print("\nscaling summary (font pt -> title pt, line unit, panel padding):")
    for pt, (title_pt, line, pad) in observed.items():
        print(f"  {pt:5g}pt -> title {title_pt:5g}pt   line {line:3d}px   pad {pad:3d}px")

    smallest, largest = FONT_SIZES[0], FONT_SIZES[-1]
    check(
        "line unit grows with font size",
        observed[largest][1] > observed[smallest][1] * 1.5,
        f"{observed[smallest][1]}px -> {observed[largest][1]}px",
    )
    check(
        "padding grows with font size",
        observed[largest][2] > observed[smallest][2],
        f"{observed[smallest][2]}px -> {observed[largest][2]}px",
    )
    check(
        "title font grows with font size",
        observed[largest][0] > observed[smallest][0] * 1.5,
        f"{observed[smallest][0]:g}pt -> {observed[largest][0]:g}pt",
    )

    print("\nHIDPI CHECK", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
