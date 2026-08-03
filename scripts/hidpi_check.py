"""Verify the UI scales with the application font instead of fixed pixels.

Builds the panels at several font sizes and checks that text still fits, that
fonts actually grow, and that spacing scales with them.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/hidpi_check.py
"""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics, QGuiApplication
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
from voicestudio.core.pronounce import PronunciationBook, PronunciationRule  # noqa: E402
from voicestudio.ui.cast_panel import CastPanel  # noqa: E402
from voicestudio.ui.design_panel import DesignPanel  # noqa: E402
from voicestudio.ui.pronounce_window import PronunciationWindow  # noqa: E402
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

    cast = CastPanel(library)
    cast.set_speakers(["NARRATOR", "VILLAIN"])

    # Expand the collapsible section so its controls are measured too.
    params.subtalker_toggle.setChecked(True)
    params.subtalker_link.setChecked(False)

    root = QWidget()
    row = QHBoxLayout(root)
    center = QVBoxLayout()
    center.addWidget(script)
    center.addWidget(cast)
    center.addWidget(params)
    center.addWidget(player)
    row.addWidget(design, 3)
    row.addLayout(center, 4)
    row.addWidget(takes, 3)
    return root, {"design": design, "script": script, "params": params,
                  "takes": takes, "cast": cast}


def build_pronunciation(app) -> PronunciationWindow:
    book = PronunciationBook()
    book.save = lambda: None  # keep the check off disk
    book.add(PronunciationRule(match="Qwen", replacement="Chwen"))
    window = PronunciationWindow(
        book,
        languages=lambda: ["Auto", "English", "German"],
        current_language=lambda: "English",
        sample_text=lambda: "Qwen has 23 voices.",
    )
    return window


def clipped(root: QWidget) -> list[str]:
    """Controls given less space than their style says they need.

    Checks both axes: width clipping truncates text, and height clipping is
    how table cell widgets (the cast dropdowns) end up spilling over their
    rows at large scale factors — the table sizes rows from the bare style,
    not from the QSS the widgets are actually rendered with.
    """
    bad = []
    for widget in root.findChildren((QPushButton, QComboBox)):
        if not widget.isVisible():
            continue
        text = widget.text() if isinstance(widget, QPushButton) else widget.currentText()
        if widget.sizeHint().width() > widget.width() + 1:
            bad.append(f"{text!r} wants {widget.sizeHint().width()}px wide, "
                       f"got {widget.width()}px")
        want_h = max(widget.sizeHint().height(), widget.minimumSizeHint().height())
        if want_h > widget.height() + 1:
            bad.append(f"{text!r} wants {want_h}px tall, got {widget.height()}px")
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

    from voicestudio.ui.fonts import FAMILY, install_application_font  # noqa: E402

    check("bundled IBM Plex Sans loads", install_application_font(app) == FAMILY)

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

        # Platform themes pin per-class fonts (QPushButton, QCheckBox, QLabel,
        # …) that ignore the application font on X11; the base QWidget rule
        # must therefore pin font-size to the application point size. Assert
        # on the stylesheet text — offscreen has no class fonts, so a missing
        # rule would render correctly here while breaking on real X11.
        check(
            "base QSS pins the application font size",
            f"font-size: {m.pt(1.0)}pt" in app.styleSheet(),
        )
        check(
            "base QSS pins the bundled font family",
            f'font-family: "{FAMILY}"' in app.styleSheet(),
        )

        # Combo popups must follow the application font — Qt's default popup
        # paints via the platform menu delegate, whose font ignores the scaled
        # app font entirely (see ThemedComboBox).
        lang = panels["script"].language_combo
        if lang.count() == 0:
            panels["script"].set_languages(["Auto", "English", "German"])
        lang.showPopup()
        app.processEvents()
        popup = lang.view()
        line_height = QFontMetrics(app.font()).height()
        check(
            "combo popup follows the application font",
            abs(popup.font().pointSizeF() - pt) < 0.6
            and popup.sizeHintForRow(0) >= line_height,
            f"view {popup.font().pointSizeF():g}pt vs {pt:g}pt, "
            f"row {popup.sizeHintForRow(0)}px vs line {line_height}px",
        )
        lang.hidePopup()

        # The pronunciation window is its own top-level window.
        pron = build_pronunciation(app)
        pron.show()
        app.processEvents()
        bad_pron = clipped(pron)
        check("pronunciation window has no clipped controls", not bad_pron,
              "; ".join(bad_pron[:3]) if bad_pron else "")
        check("pronunciation window scales with the font",
              pron.width() >= m.ch(100), f"{pron.width()}px wide")
        pron.close()

        # The window must stay shrinkable well below its default size, or
        # window managers refuse to maximize on screens the default barely
        # fits — they honour min-size hints, and the unwrapped design column
        # once pushed the minimum past the work area (the reason it is
        # scroll-wrapped). 0.92 of the default height catches that state.
        from voicestudio.ui.main_window import MainWindow  # noqa: E402

        MainWindow._start_engine = lambda self: None  # UI only, no model
        win = MainWindow()
        win.cast_panel.setVisible(True)  # worst case: every row present
        win.show()
        app.processEvents()
        min_h = win.minimumSizeHint().height()
        budget = m.line * 18  # scroll-wrapped columns keep it near ~13 lines
        check("window minimum height leaves room to maximize",
              min_h <= budget, f"min {min_h}px vs budget {budget}px")
        win.hide()
        win.deleteLater()
        app.processEvents()
        pron.deleteLater()

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
