"""Headless GUI test: build the window, load the model, generate, verify.

Run:  QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/gui_smoke.py
"""

import sys
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicestudio.ui.main_window import MainWindow  # noqa: E402
from voicestudio.ui.metrics import refresh  # noqa: E402
from voicestudio.ui.theme import build_qss  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def wait_for(signal, timeout_ms: int) -> bool:
    """Block the test (not the UI) until `signal` fires or we time out."""
    loop = QEventLoop()
    fired = {"yes": False}

    def on_fire(*_):
        fired["yes"] = True
        loop.quit()

    signal.connect(on_fire)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    try:
        signal.disconnect(on_fire)
    except TypeError:
        pass
    return fired["yes"]


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(build_qss(refresh()))

    window = MainWindow()
    window.show()

    check("window constructs", window.isVisible())
    check("20 templates loaded", len(window.templates.templates) == 20)
    check(
        "generate disabled while loading",
        not window.script_panel.generate_button.isEnabled(),
    )
    # The composer must stay usable during the cold load.
    window.script_panel.text_edit.setPlainText("Testing the studio.")
    check("script editable during load", window.script_panel.script == "Testing the studio.")

    check("model loaded", wait_for(window.host.loadReady, 180_000))
    app.processEvents()
    check("generate enabled after load", window.script_panel.generate_button.isEnabled())
    check("languages populated", window.script_panel.language_combo.count() > 5,
          f"{window.script_panel.language_combo.count()} languages")

    # Pick a template, which should fill the instruct box and traits.
    template = window.templates.get("nature-documentary")
    window.design_panel._apply_traits(template.traits)
    window.design_panel.set_instruct(template.instruct, detached=False)
    check("template fills description", window.design_panel.instruct == template.instruct)

    before = len(window.history.all())
    window.script_panel.language_combo.setCurrentText("English")
    window.generate_one()
    check("single take generated", wait_for(window.host.jobDone, 300_000))
    app.processEvents()

    takes = window.history.all()
    check("history grew by one", len(takes) == before + 1, f"{before} -> {len(takes)}")
    if takes:
        newest = takes[0]
        check("wav written", Path(newest.wav_path).is_file())
        check("audio has duration", newest.duration > 0.5, f"{newest.duration:.2f}s")
        check("take card rendered", window.takes_panel.cards_layout.count() - 1 == len(takes))

        # A/B pinning
        window.takes_panel._assign_slot(newest.id, "A")
        check("slot A pinned", window.takes_panel.slot_a == newest.id)
        window.takes_panel._assign_slot(newest.id, "B")
        check(
            "same take cannot hold both slots",
            window.takes_panel.slot_b == newest.id and window.takes_panel.slot_a is None,
        )

        # Restore params round-trip
        window._restore_params(newest.id)
        check("params restored", window.script_panel.script == newest.text)

    # Saving a voice must capture the seed that produced what you just heard.
    check(
        "save dialog can read the current seed",
        window.design_panel.seed_provider is not None
        and window.design_panel.seed_provider() == window.params_panel.seed_spin.value(),
    )
    from voicestudio.core.library import VoicePreset  # noqa: E402

    pinned = window.library.add(
        VoicePreset(name="__gui_probe__", instruct="a test voice", seed=4242)
    )
    window.design_panel.refresh_library()
    window.params_panel.seed_lock.setChecked(False)
    window.design_panel.load_preset(pinned.id)
    app.processEvents()
    check("loading a pinned preset restores its seed",
          window.params_panel.seed_spin.value() == 4242)
    check("loading a pinned preset locks the seed",
          window.params_panel.seed_lock.isChecked())
    window.library.remove(pinned.id)
    window.design_panel.refresh_library()

    # Variations: distinct seeds
    before = len(window.history.all())
    window.generate_variations(2)
    check("variations completed", wait_for(window.host.jobDone, 600_000))
    app.processEvents()
    after = window.history.all()
    new_takes = after[: len(after) - before]
    check("two variations added", len(after) == before + 2, f"{before} -> {len(after)}")
    check(
        "variation seeds differ",
        len({t.seed for t in new_takes}) == len(new_takes),
        str([t.seed for t in new_takes]),
    )

    # Long script chunking
    long_text = " ".join(["This is sentence number %d." % i for i in range(40)])
    window.script_panel.text_edit.setPlainText(long_text)
    check("long script chunks", len(window.script_panel.chunks()) > 1,
          f"{len(window.script_panel.chunks())} chunks")

    window.close()
    print("\nGUI SMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
