"""Render the studio to docs/screenshot.png (loads model, makes one real take).

Usage:  screenshot.py [--pt POINT_SIZE] [--out PATH]
`--pt` renders at a different application font size, which is how HiDPI
desktops scale — useful for checking the layout holds up.
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QEventLoop, QTimer
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.ui.main_window import MainWindow  # noqa: E402
from voicestudio.ui.metrics import refresh  # noqa: E402
from voicestudio.ui.theme import build_qss  # noqa: E402


def wait_for(signal, timeout_ms):
    loop = QEventLoop()
    signal.connect(lambda *_: loop.quit())
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


def settle(app, ms=400):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=float, default=0.0, help="application font point size")
    parser.add_argument("--out", default=str(ROOT / "docs" / "screenshot.png"))
    args = parser.parse_args()

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    if args.pt > 0:
        font = QFont(app.font())
        font.setPointSizeF(args.pt)
        app.setFont(font)
    app.setStyleSheet(build_qss(refresh()))

    window = MainWindow()
    window.show()

    wait_for(window.host.loadReady, 180_000)
    settle(app)

    template = window.templates.get("noir-detective")
    window.design_panel._apply_traits(template.traits)
    window.design_panel.set_instruct(template.instruct, detached=False)
    window.design_panel.template_search.setText("noir")
    window.script_panel.language_combo.setCurrentText("English")
    window.script_panel.text_edit.setPlainText(
        "She walked in like a rumor that turned out to be true. "
        "I had two questions and neither one was polite."
    )
    settle(app)

    window.generate_one()
    wait_for(window.host.jobDone, 300_000)
    settle(app, 800)

    takes = window.history.all()
    if takes:
        window.takes_panel._assign_slot(takes[0].id, "A")
        if len(takes) > 1:
            window.takes_panel._assign_slot(takes[1].id, "B")
        window.history.toggle_star(takes[0].id)
        window.takes_panel.refresh()
    settle(app)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = window.grab().save(str(out))
    print(f"{'saved' if ok else 'FAILED to save'} {out}")

    window.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
