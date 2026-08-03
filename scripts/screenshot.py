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

    from voicestudio.ui.fonts import install_application_font  # noqa: E402

    install_application_font(app)
    if args.pt > 0:
        font = QFont(app.font())
        font.setPointSizeF(args.pt)
        app.setFont(font)
    app.setStyleSheet(build_qss(refresh()))

    window = MainWindow()
    window.show()

    wait_for(window.host.loadReady, 180_000)
    settle(app)

    templates = window.templates
    noir = templates.get("noir-detective")
    window.design_panel._apply_traits(noir.traits)
    window.design_panel.set_instruct(noir.instruct, detached=False)
    window.design_panel.template_search.setText("noir")
    window.script_panel.language_combo.setCurrentText("English")

    # A dialogue script, so the cast table is on screen.
    window.script_panel.text_edit.setPlainText(
        "NARRATOR: She walked in like a rumor that turned out to be true.\n"
        "DETECTIVE: I had two questions and neither one was polite.\n"
        "[pause 1.0]\n"
        "NARRATOR: Nobody answered.\n"
    )
    settle(app)

    # A saved voice, so the cast shows a custom entry beside the templates.
    from voicestudio.core.library import VoicePreset

    saved = window.library.add(VoicePreset(
        name="Rain-Slick Detective", instruct=noir.instruct,
        traits=dict(noir.traits), seed=4242, favorite=True,
    ))
    window.design_panel.refresh_library()
    window.cast_panel.set_cast({
        "NARRATOR": f"template:{templates.get('nature-documentary').id}",
        "DETECTIVE": f"preset:{saved.id}",
    })
    window.cast_panel.set_speakers(["NARRATOR", "DETECTIVE"])
    window.params_panel.subtalker_toggle.setChecked(True)
    settle(app)

    window.generate_dialogue()
    wait_for(window.host.jobDone, 600_000)
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

    # Second shot: the pronunciation window.
    from voicestudio.core.pronounce import PronunciationRule

    existing = {r.match for r in window.book.all()}
    added = []
    for match, replacement, note in [
        ("Qwen", "Chwen", "model name"),
        ("Caius", "Kye-us", "character name"),
    ]:
        if match not in existing:
            added.append(window.book.add(PronunciationRule(
                match=match, replacement=replacement, note=note)))

    window.script_panel.text_edit.setPlainText(
        "Qwen shipped 23 voices on 2026-07-26, and Caius paid $5.50 for the 3rd one."
    )
    window.open_pronunciation()
    settle(app, 500)
    pron = window.pronounce_window
    pron.table.selectRow(0)
    pron.refresh_preview()
    settle(app, 300)

    pron_out = out.parent / "pronunciation.png"
    ok2 = pron.grab().save(str(pron_out))
    print(f"{'saved' if ok2 else 'FAILED to save'} {pron_out}")

    for rule in added:
        window.book.remove(rule.id)
    window.library.remove(saved.id)
    pron.close()
    window.close()
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
