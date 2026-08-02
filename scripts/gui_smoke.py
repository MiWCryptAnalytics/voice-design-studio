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

    # Voice profile: UI wiring and request routing. Uses a synthetic embedding
    # so none of this needs the Base checkpoint.
    import numpy as np  # noqa: E402

    from voicestudio.engine import VoiceProfile  # noqa: E402

    dp = window.design_panel
    check("profile controls exist",
          hasattr(dp, "drop_zone") and dp.import_profile_button.isEnabled()
          and dp.extract_profile_button.isEnabled())
    check("no profile at start", window.voice_profile is None)
    check("clear disabled without a profile", not dp.clear_profile_button.isEnabled())

    fake = VoiceProfile(name="__probe__", embedding=np.ones(192, dtype=np.float32))
    window.voice_profile = fake
    dp.set_profile_status(fake.name)
    check("profile status shows the name", "__probe__" in dp.profile_label.text())
    check("clear enabled with a profile", dp.clear_profile_button.isEnabled())

    window.params_panel.temperature.setValue(0.9)
    probe_request = window._build_request("Testing.", 1234)
    check("profile rides on the request", probe_request.voice_profile is fake)
    check("clone temperature capped at 0.7",
          abs(probe_request.temperature - 0.7) < 1e-9, str(probe_request.temperature))
    check("sub-talker follows the cap while linked",
          abs(probe_request.subtalker_temperature - 0.7) < 1e-9)
    probe_request = window._build_request("Testing.", 1234, use_profile=False)
    check("profile can be bypassed",
          probe_request.voice_profile is None and probe_request.temperature == 0.9)

    window._clear_profile()
    check("clear drops the profile", window.voice_profile is None)
    check("cleared profile empties the setting", window.settings.voice_profile == "")
    check("status returns to description mode",
          not dp.clear_profile_button.isEnabled())

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

    # Long script chunking. The limit is 1500 chars now, so a script has to be
    # genuinely long before it needs splitting at all.
    medium = " ".join(["This is sentence number %d." % i for i in range(30)])
    window.script_panel.text_edit.setPlainText(medium)
    check("a medium script stays in one pass (no seams)",
          len(window.script_panel.chunks()) == 1,
          f"{len(medium)} chars -> {len(window.script_panel.chunks())} chunk")

    long_text = " ".join(["This is sentence number %d." % i for i in range(90)])
    window.script_panel.text_edit.setPlainText(long_text)
    check("a genuinely long script still chunks",
          len(window.script_panel.chunks()) > 1,
          f"{len(long_text)} chars -> {len(window.script_panel.chunks())} chunks")
    check("chunks respect the limit",
          max(len(c) for c in window.script_panel.chunks()) <= 1500)

    # ---- sub-talker ----
    params = window.params_panel
    params.subtalker_link.setChecked(True)
    params.temperature.setValue(0.7)
    app.processEvents()
    check("linked sub-talker mirrors the main controls",
          params.values()["subtalker_temperature"] == 0.7)

    params.subtalker_link.setChecked(False)
    params.subtalker_temperature.setValue(1.3)
    app.processEvents()
    check("unlinked sub-talker holds its own value",
          params.values()["subtalker_temperature"] == 1.3
          and params.values()["temperature"] == 0.7)
    check("sub-talker controls disabled while linked",
          not params.subtalker_temperature.isEnabled()
          if params.subtalker_link.isChecked() else True)
    params.reset_defaults()
    check("reset relinks and restores model defaults",
          params.subtalker_link.isChecked() and params.values()["max_new_tokens"] == 8192)

    # ---- pronunciation ----
    from voicestudio.core.pronounce import PronunciationRule  # noqa: E402

    window.script_panel.text_edit.setPlainText("Qwen has 23 voices.")
    rule = window.book.add(PronunciationRule(match="Qwen", replacement="Chwen"))
    spoken = window.spoken(window.script_panel.script)
    check("rules and normalization both reach the spoken text",
          "Chwen" in spoken.text and "twenty-three" in spoken.text, spoken.text)
    check("request carries the spoken text, not the raw text",
          "Chwen" in window._build_request(window.script_panel.script, 1).text)

    window.open_pronunciation()
    app.processEvents()
    pw = window.pronounce_window
    check("pronunciation window opens", pw is not None and pw.isVisible())
    check("window lists the rule", pw.table.rowCount() == len(window.book.all()))
    pw.refresh_preview()
    check("preview shows the transformed text", "Chwen" in pw.preview_out.toPlainText())

    window.book.enabled = False
    check("disabling the book passes text through untouched",
          window.spoken("Qwen has 23 voices.").text == "Qwen has 23 voices.")
    window.book.enabled = True
    window.book.remove(rule.id)
    pw.close()

    # ---- dialogue ----
    # The cast persists across sessions, so start from a known-empty one.
    window.cast_panel.set_cast({})
    window.script_panel.text_edit.setPlainText(
        "NARRATOR: The door opened.\n"
        "VILLAIN: You're late.\n"
        "NARRATOR: Nobody answered.\n"
    )
    app.processEvents()
    check("dialogue detected", window.script_panel.dialogue_script() is not None)
    check("cast panel appears for dialogue", window.cast_panel.isVisible())
    check("cast lists both speakers",
          window.cast_panel._speakers == ["NARRATOR", "VILLAIN"],
          str(window.cast_panel._speakers))
    check("unassigned speakers reported",
          sorted(window.cast_panel.unassigned()) == ["NARRATOR", "VILLAIN"])

    before = len(window.history.all())
    window.generate_dialogue()  # cast incomplete: must refuse without generating
    app.processEvents()
    check("incomplete cast blocks generation", len(window.history.all()) == before)

    # A voice saved while the cast is on screen must appear without a reload.
    def cast_combo_texts():
        combo = window.cast_panel.table.cellWidget(0, 1)
        return [combo.itemText(i) for i in range(combo.count())]

    check("custom voice absent before saving",
          not any("Gravel Guy" in t for t in cast_combo_texts()))
    custom = window.library.add(
        VoicePreset(name="Gravel Guy", instruct="a gravelly voice", seed=4242)
    )
    window.design_panel.libraryChanged.emit()
    app.processEvents()
    check("custom voice appears in the cast immediately",
          any("Gravel Guy" in t for t in cast_combo_texts()))
    check("pinned custom voice is marked in the cast",
          any("Gravel Guy" in t and "📌" in t for t in cast_combo_texts()))
    check("cast groups my voices separately from templates",
          any("— My voices —" in t for t in cast_combo_texts())
          and any("— Templates —" in t for t in cast_combo_texts()))

    window.cast_panel.set_cast({"NARRATOR": f"preset:{custom.id}"})
    window.cast_panel.refresh_voices()
    check("custom voice is assignable",
          window.cast_panel.instruct_for(f"preset:{custom.id}") == "a gravelly voice")

    # Deleting an assigned voice must unassign it, not silently substitute another.
    window.library.remove(custom.id)
    window.design_panel.libraryChanged.emit()
    app.processEvents()
    check("deleting an assigned voice unassigns that speaker",
          "NARRATOR" in window.cast_panel.unassigned())

    templates = window.templates
    window.cast_panel.set_cast({
        "NARRATOR": f"template:{templates.get('nature-documentary').id}",
        "VILLAIN": f"template:{templates.get('sinister-villain').id}",
    })
    window.cast_panel.set_speakers(["NARRATOR", "VILLAIN"])
    check("cast complete after assignment", not window.cast_panel.unassigned())

    check("voice locking is on by default", window.cast_panel.lock_voices.isChecked())
    window.generate_dialogue()
    check("dialogue generated", wait_for(window.host.jobDone, 600_000))
    app.processEvents()
    takes = window.history.all()
    check("dialogue produced one joined take", len(takes) == before + 1,
          f"{before} -> {len(takes)}")
    check("segments retained for stem export",
          window._last_segments is not None and len(window._last_segments[1]) == 3,
          str(len(window._last_segments[1]) if window._last_segments else 0))
    if window._last_segments:
        segments = window._last_segments[1]
        check("segments carry their speakers",
              [s.speaker for s in segments] == ["NARRATOR", "VILLAIN", "NARRATOR"])
        joined = takes[0].duration
        parts = sum(s.duration + s.gap_after for s in segments)
        check("joined duration matches segments plus gaps",
              abs(joined - parts) < 0.05, f"{joined:.2f}s vs {parts:.2f}s")

    # Unlocked dialogue must still work — it's the per-line fallback path.
    before = len(window.history.all())
    window.cast_panel.lock_voices.setChecked(False)
    window.generate_dialogue()
    check("unlocked dialogue still generates", wait_for(window.host.jobDone, 600_000))
    app.processEvents()
    check("unlocked run produced a take", len(window.history.all()) == before + 1)
    check("unlocked run still yields per-line segments",
          window._last_segments is not None and len(window._last_segments[1]) == 3)
    window.cast_panel.lock_voices.setChecked(True)

    window.close()
    print("\nGUI SMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
