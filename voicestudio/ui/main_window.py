"""Main studio window: layout, engine thread wiring, and the generate loop."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..core import audio as audio_utils
from ..core import dialogue as dialogue_mod
from ..core.config import EXPORTS_DIR, PROFILES_DIR, Settings, ensure_dirs
from ..core.history import Take, TakeHistory
from ..core.library import VoiceLibrary
from ..core.pronounce import PronunciationBook, SpokenText
from ..core.templates import load_templates
from ..engine import SynthRequest, VoiceProfile, new_seed
from .cast_panel import CastPanel
from .design_panel import DesignPanel
from .metrics import metrics
from .params_panel import ParamsPanel
from .player import PlayerBar
from .pronounce_window import PronunciationWindow
from .script_panel import ScriptPanel
from .takes_panel import TakesPanel
from .workers import (
    COMPARE,
    DIALOGUE,
    SCRIPT,
    SCRIPT_JOIN_GAP,
    SINGLE,
    VARIATIONS,
    EngineHost,
    JobItem,
    SynthJob,
    TakeAudio,
)


# With a fixed speaker embedding the voice can't drift, but the talker still
# samples rhythm freely — capping its temperature a little below the 0.9
# default keeps pacing consistent from line to line.
CLONE_TEMPERATURE = 0.7


class MainWindow(QMainWindow):
    loadRequested = pyqtSignal()
    jobRequested = pyqtSignal(object)
    profileRequested = pyqtSignal(str, str)  # wav path, display name

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()

        self.settings = Settings.load()
        self.library = VoiceLibrary.load()
        self.history = TakeHistory.load()
        self.templates = load_templates()
        self.book = PronunciationBook.load()

        self._busy = False
        self._autoplay_next = False
        self._pending_takes = 0
        self.pronounce_window: PronunciationWindow | None = None
        # When set, the next two takes get pinned into the A/B slots.
        self._ab_capture: list[str] | None = None
        self._last_segments: tuple | None = None
        # The active fixed-embedding voice; None means text-description mode.
        self.voice_profile: VoiceProfile | None = None
        self._profile_path: Path | None = None
        # Golden-sample flow: name to give the take being captured, then the
        # (wav, name) pair waiting for extraction once the job finishes.
        self._capture_profile: str | None = None
        self._profile_pending: tuple[str, str] | None = None

        self.setWindowTitle(APP_NAME)
        self.resize(*self._default_size())
        self._build_ui()
        self._build_shortcuts()
        self._restore_settings()
        self._start_engine()

    # ---------- layout ----------

    def _default_size(self) -> tuple[int, int]:
        """Size from the font, then clamp to the screen it will open on."""
        m = metrics()
        width, height = m.ch(185), m.sp(52)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.94))
            height = min(height, int(available.height() * 0.92))
        return width, height

    def _build_ui(self) -> None:
        self.design_panel = DesignPanel(self.library)
        self.design_panel.auditionRequested.connect(self._audition)
        self.design_panel.seedApplied.connect(self._apply_pinned_seed)
        # Saving or deleting a voice must show up in the cast immediately.
        self.design_panel.libraryChanged.connect(self._on_library_changed)
        self.design_panel.profileImportRequested.connect(self._import_profile)
        self.design_panel.profileExtractRequested.connect(
            self._extract_profile_from_description
        )
        self.design_panel.profileCleared.connect(self._clear_profile)

        self.script_panel = ScriptPanel()
        self.script_panel.generateRequested.connect(self.generate_one)
        self.script_panel.variationsRequested.connect(self.generate_variations)
        self.script_panel.cancelRequested.connect(self._cancel_job)
        self.script_panel.dialogueDetected.connect(self._on_dialogue_detected)

        self.cast_panel = CastPanel(self.library)
        self.cast_panel.auditionRequested.connect(self._audition)
        self.cast_panel.castChanged.connect(self._on_cast_changed)
        self.cast_panel.setVisible(False)

        self.params_panel = ParamsPanel()
        # Saving a voice needs the seed that produced what you last heard.
        self.design_panel.seed_provider = lambda: self.params_panel.seed_spin.value()
        self.player = PlayerBar()

        player_frame = QFrame()
        player_frame.setProperty("role", "panel")
        m = metrics()
        player_layout = QVBoxLayout(player_frame)
        pad = m.sp(0.7)
        player_layout.setContentsMargins(pad, pad, pad, pad)
        player_layout.setSpacing(m.sp(0.15))
        player_title = QLabel("PREVIEW")
        player_title.setProperty("role", "title")
        player_layout.addWidget(player_title)
        player_layout.addWidget(self.player)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(m.sp(0.15))
        center_layout.addWidget(self.script_panel, 3)
        center_layout.addWidget(self.cast_panel, 0)
        center_layout.addWidget(self.params_panel, 0)
        center_layout.addWidget(player_frame, 0)

        self.takes_panel = TakesPanel(self.history)
        self.takes_panel.playRequested.connect(self._play_take)
        self.takes_panel.rerollRequested.connect(self._reroll)
        self.takes_panel.restoreRequested.connect(self._restore_params)
        self.takes_panel.exportRequested.connect(self._export_take)

        # Unwrapped, the tallest column's minimum height becomes the *window's*
        # minimum height, and at large UI scales that exceeds the work area —
        # window managers honour min-size hints and then refuse to maximize.
        # Scroll-wrapping the columns keeps the window shrinkable; the
        # scrollbars only appear when the window really is too short.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(_shrinkable(self.design_panel))
        splitter.addWidget(_shrinkable(center))
        splitter.addWidget(self.takes_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)
        width = self._default_size()[0]
        splitter.setSizes(
            [int(width * 0.29), int(width * 0.42), int(width * 0.29)]
        )

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(m.sp(0.6), m.sp(0.6), m.sp(0.6), m.sp(0.35))
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self._build_menu()
        self._build_status_bar()

    def _build_menu(self) -> None:
        tools = self.menuBar().addMenu("&Tools")
        action = tools.addAction("&Pronunciation…")
        action.setShortcut(QKeySequence("Ctrl+P"))
        action.triggered.connect(self.open_pronunciation)

        tools.addSeparator()
        stems = tools.addAction("Export dialogue &stems…")
        stems.setToolTip("Write one WAV per line plus a cue sheet")
        stems.triggered.connect(self._export_stems_of_latest)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()

        self.status_label = QLabel("Starting…")
        bar.addWidget(self.status_label, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(metrics().ch(22))
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

        self.perf_label = QLabel("")
        self.perf_label.setProperty("role", "metric")
        bar.addPermanentWidget(self.perf_label)

        self.device_label = QLabel("")
        self.device_label.setProperty("role", "metric")
        bar.addPermanentWidget(self.device_label)

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram)
        self._vram_timer.start(2000)

    def _build_shortcuts(self) -> None:
        def bind(sequence: str, handler) -> None:
            QShortcut(QKeySequence(sequence), self, activated=handler)

        bind("Ctrl+Return", self.generate_one)
        bind("Ctrl+Enter", self.generate_one)
        bind("Ctrl+S", self.design_panel.save_current_as_preset)
        bind("Ctrl+1", lambda: self.takes_panel.play_slot("A"))
        bind("Ctrl+2", lambda: self.takes_panel.play_slot("B"))
        bind("Space", self._space_toggle)
        bind("Escape", self._cancel_job)

    def _space_toggle(self) -> None:
        """Space plays/pauses unless the user is typing."""
        focus = self.focusWidget()
        if focus is not None and focus.metaObject().className() in (
            "QPlainTextEdit",
            "QLineEdit",
            "QTextEdit",
            "QSpinBox",
            "QDoubleSpinBox",
        ):
            return
        self.player.toggle()

    # ---------- engine thread ----------

    def _start_engine(self) -> None:
        self.thread = QThread(self)
        self.host = EngineHost()
        self.host.moveToThread(self.thread)

        self.loadRequested.connect(self.host.load)
        self.jobRequested.connect(self.host.run_job)
        self.profileRequested.connect(self.host.extract_profile)

        self.host.loadProgress.connect(self._on_load_progress)
        self.host.loadReady.connect(self._on_load_ready)
        self.host.loadFailed.connect(self._on_load_failed)
        self.host.jobStarted.connect(self._on_job_started)
        self.host.jobProgress.connect(self._on_job_progress)
        self.host.takeReady.connect(self._on_take_ready)
        self.host.jobDone.connect(self._on_job_done)
        self.host.jobFailed.connect(self._on_job_failed)
        self.host.profileReady.connect(self._on_profile_ready)
        self.host.profileFailed.connect(self._on_profile_failed)

        self.thread.start()
        self.script_panel.set_busy(True)
        self.script_panel.cancel_button.setVisible(False)
        self.loadRequested.emit()

    def _on_load_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_load_ready(self, loaded) -> None:
        self.script_panel.set_languages(loaded.languages)
        if self.settings.language in loaded.languages:
            self.script_panel.language_combo.setCurrentText(self.settings.language)
        self.status_label.setText(
            "Ready · loaded from local cache" if loaded.offline else "Ready"
        )
        source = "cached" if loaded.offline else "hub"
        self.device_label.setText(
            f"{loaded.device_label} · {loaded.sample_rate} Hz · {source}"
        )
        self.device_label.setToolTip(
            "Model loaded from the local cache — no network needed."
            if loaded.offline
            else "Model was fetched from the Hugging Face Hub this launch."
        )
        self.script_panel.set_busy(False)

    def _on_load_failed(self, message: str) -> None:
        self.status_label.setText("Model failed to load")
        QMessageBox.critical(
            self,
            "Could not load model",
            f"{message}\n\nThe app will stay open, but generation is unavailable.",
        )

    # ---------- generating ----------

    def spoken(self, text: str) -> SpokenText:
        """What will actually be spoken, after normalization and rules."""
        return self.book.apply(text, self.script_panel.language)

    def _build_request(
        self, text: str, seed: int, use_profile: bool = True
    ) -> SynthRequest:
        params = self.params_panel.values()
        profile = self.voice_profile if use_profile else None
        if profile is not None:
            # Identity is pinned by the embedding; a slightly cooler talker
            # keeps rhythm consistent across the batch. The sub-talker follows
            # only when it is mirroring the main value anyway.
            capped = min(params["temperature"], CLONE_TEMPERATURE)
            if params["subtalker_temperature"] == params["temperature"]:
                params["subtalker_temperature"] = capped
            params["temperature"] = capped
        return SynthRequest(
            text=self.spoken(text).text,
            instruct=self.design_panel.instruct,
            language=self.script_panel.language,
            seed=seed,
            voice_profile=profile,
            **params,
        )

    def _submit(self, job: SynthJob, autoplay: bool = False) -> None:
        if self._busy:
            return
        if not self.host.is_ready:
            self.status_label.setText("Model is still loading…")
            return
        self._busy = True
        self._autoplay_next = autoplay
        self.script_panel.set_busy(True)
        self.jobRequested.emit(job)

    def generate_one(self) -> None:
        if self.script_panel.dialogue_script() is not None:
            self.generate_dialogue()
            return

        chunks = self.script_panel.chunks()
        if not chunks:
            self.status_label.setText("Write something for the voice to say.")
            return
        seed = self.params_panel.effective_seed()
        request = self._build_request(chunks[0], seed)
        items = [JobItem(text=self.spoken(c).text) for c in chunks]
        mode = SCRIPT if len(chunks) > 1 else SINGLE
        self._submit(
            SynthJob(
                request=request,
                items=items,
                seeds=[seed],
                mode=mode,
                gap_seconds=SCRIPT_JOIN_GAP,
            ),
            autoplay=True,
        )

    def generate_dialogue(self) -> None:
        script = self.script_panel.dialogue_script()
        if script is None or not script.lines:
            self.status_label.setText("Nothing to generate.")
            return

        missing = self.cast_panel.unassigned()
        if missing:
            # Refuse before generating, not halfway through a long script. Reported
            # inline rather than in a modal — the cast table already shows which
            # speakers are unassigned, and a dialog here would just be in the way.
            self.status_label.setText(
                "Assign a voice to every speaker first — unassigned: "
                + ", ".join(missing)
            )
            self.cast_panel.setFocus()
            return

        cast = self.cast_panel.cast
        fallback = self.design_panel.instruct
        seed = self.params_panel.effective_seed()

        items = [
            JobItem(
                text=self.spoken(line.text).text,
                instruct=self.cast_panel.instruct_for(cast.get(line.speaker, ""),
                                                      fallback),
                speaker=line.speaker,
                pause_after=line.pause_after,
            )
            for line in script.lines
        ]
        # Dialogue keeps its per-speaker cast voices — a single fixed embedding
        # would collapse every character into the same speaker.
        request = self._build_request(script.lines[0].text, seed, use_profile=False)
        self._submit(
            SynthJob(
                request=request,
                items=items,
                seeds=[seed],
                mode=DIALOGUE,
                lock_voices=self.cast_panel.lock_voices.isChecked(),
            ),
            autoplay=True,
        )

    def generate_variations(self, count: int) -> None:
        chunks = self.script_panel.chunks()
        if not chunks:
            self.status_label.setText("Write something for the voice to say.")
            return
        # Variations explore seed space, so the lock never applies here.
        seeds = [new_seed() for _ in range(count)]
        request = self._build_request(chunks[0], seeds[0])
        self._submit(
            SynthJob(
                request=request,
                items=[JobItem(text=request.text)],
                seeds=seeds,
                mode=VARIATIONS,
            )
        )

    def _audition(self, instruct: str, name: str) -> None:
        """Generate a template against the fixed demo line and seed."""
        request = SynthRequest(
            text=self.templates.demo_sentence,
            instruct=instruct,
            language=self.script_panel.language,
            seed=self.templates.demo_seed,
            **self.params_panel.values(),
        )
        self._submit(
            SynthJob(
                request=request,
                items=[JobItem(text=self.templates.demo_sentence)],
                seeds=[self.templates.demo_seed],
                mode=SINGLE,
                label=f"Audition · {name}",
            ),
            autoplay=True,
        )

    # ---------- voice profile ----------

    def _import_profile(self, path: str) -> None:
        """Clone from a user-supplied reference recording."""
        if self._busy:
            self.status_label.setText("Busy — try again after the current job.")
            return
        self._request_profile_extraction(path, Path(path).stem)

    def _extract_profile_from_description(self) -> None:
        """Generate a golden sample of the description, then lock its speaker.

        The sample is rendered by the VoiceDesign checkpoint as a normal take
        (so it can be auditioned and kept); once the job finishes its WAV is
        handed to the speaker encoder and the embedding becomes the profile.
        """
        if self._busy:
            self.status_label.setText("Busy — try again after the current job.")
            return
        name = _profile_name(self.design_panel.instruct)
        seed = self.params_panel.effective_seed()
        # use_profile=False: the golden sample must come from the description,
        # not from whatever profile is currently active.
        request = self._build_request(
            self.templates.demo_sentence, seed, use_profile=False
        )
        self._capture_profile = name
        self._submit(
            SynthJob(
                request=request,
                items=[JobItem(text=request.text)],
                seeds=[seed],
                mode=SINGLE,
                label=f"Golden sample · {name}",
            ),
            autoplay=True,
        )
        if not self._busy:  # _submit refused (model still loading)
            self._capture_profile = None

    def _request_profile_extraction(self, wav_path: str, name: str) -> None:
        self._busy = True
        self.script_panel.set_busy(True)
        self.design_panel.set_profile_status(None, busy=True)
        self.status_label.setText(
            "Extracting voice profile… (first use loads the clone model)"
        )
        self.profileRequested.emit(wav_path, name)

    def _on_profile_ready(self, profile: VoiceProfile) -> None:
        self._busy = False
        self.script_panel.set_busy(False)
        self.voice_profile = profile
        self._profile_path = None
        try:
            target = PROFILES_DIR / f"{uuid4().hex[:8]}_{_safe_name(profile.name)}.npz"
            self._profile_path = profile.save(target)
        except OSError:
            pass  # profile still works for this session, it just won't persist
        self.settings.voice_profile = (
            str(self._profile_path) if self._profile_path else ""
        )
        self.design_panel.set_profile_status(profile.name)
        self.status_label.setText(
            f"Voice profile “{profile.name}” locked — every line now shares "
            "one fixed voice"
        )

    def _on_profile_failed(self, message: str) -> None:
        self._busy = False
        self.script_panel.set_busy(False)
        self.design_panel.set_profile_status(
            self.voice_profile.name if self.voice_profile else None
        )
        self.status_label.setText("Voice profile extraction failed")
        QMessageBox.warning(self, "Voice profile", message)

    def _clear_profile(self) -> None:
        self.voice_profile = None
        self._profile_path = None
        self.settings.voice_profile = ""
        self.design_panel.set_profile_status(None)
        self.status_label.setText(
            "Voice profile cleared — the voice follows the description again"
        )

    def _on_dialogue_detected(self, script) -> None:
        """Show the cast only when the script actually uses speaker labels."""
        is_dialogue = script is not None
        self.cast_panel.setVisible(is_dialogue)
        if is_dialogue:
            self.cast_panel.set_speakers(script.speakers)
            for issue in script.issues:
                self.status_label.setText(f"Line {issue.line_number}: {issue.message}")

    def _on_cast_changed(self) -> None:
        self.settings.cast = self.cast_panel.cast

    def _on_library_changed(self) -> None:
        self.cast_panel.refresh_voices()

    # ---------- pronunciation ----------

    def open_pronunciation(self) -> None:
        if self.pronounce_window is None:
            self.pronounce_window = PronunciationWindow(
                self.book,
                languages=lambda: [
                    self.script_panel.language_combo.itemText(i)
                    for i in range(self.script_panel.language_combo.count())
                ],
                current_language=lambda: self.script_panel.language,
                sample_text=lambda: self.script_panel.script,
                parent=self,
            )
            self.pronounce_window.changed.connect(self._on_pronunciation_changed)
            self.pronounce_window.testRequested.connect(self._test_pronunciation)
        self.pronounce_window.reload()
        self.pronounce_window.show()
        self.pronounce_window.raise_()
        self.pronounce_window.activateWindow()

    def _on_pronunciation_changed(self) -> None:
        spoken = self.spoken(self.script_panel.script)
        if spoken.changed:
            self.status_label.setText(
                f"Pronunciation active · {spoken.total_rule_hits} rule matches"
            )

    def _test_pronunciation(self, rule_id: str, phrase: str) -> None:
        """Generate the phrase with and without one rule, then pin them A/B."""
        language = self.script_panel.language
        with_rule = self.book.apply(phrase, language).text
        without_rule = self.book.apply(phrase, language, skip_rule=rule_id).text
        seed = self.params_panel.seed_spin.value() or new_seed()

        # Built through _build_request so the A/B pair is spoken by whichever
        # voice (profile or description) the script itself would use.
        request = replace(self._build_request(with_rule, seed), text=with_rule)
        self._ab_capture = []
        self._submit(
            SynthJob(
                request=request,
                items=[
                    JobItem(text=with_rule, label="Pronunciation · with rule"),
                    JobItem(text=without_rule, label="Pronunciation · without rule"),
                ],
                seeds=[seed],
                mode=COMPARE,
            )
        )

    def _apply_pinned_seed(self, seed: int) -> None:
        """A pinned preset must actually reproduce its voice, so lock the seed."""
        self.params_panel.show_seed(seed)
        self.params_panel.seed_lock.setChecked(True)
        self.status_label.setText(
            f"Loaded voice pinned to seed {seed} — locked to reproduce that rendition"
        )

    def _cancel_job(self) -> None:
        if self._busy:
            self.host.cancel()
            self.status_label.setText("Cancelling after the current chunk…")

    # ---------- job callbacks ----------

    def _on_job_started(self, total: int) -> None:
        self._pending_takes = total
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)

    def _on_job_progress(self, message: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(message)

    def _on_take_ready(self, take_audio: TakeAudio) -> None:
        wav_path = self.history.next_wav_path()
        audio_utils.write_wav(wav_path, take_audio.waveform, take_audio.sample_rate)

        request = take_audio.request
        take = Take(
            instruct=request.instruct,
            text=request.text,
            language=request.language,
            seed=take_audio.seed,
            wav_path=str(wav_path),
            sample_rate=take_audio.sample_rate,
            duration=take_audio.duration,
            elapsed=take_audio.elapsed,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_new_tokens,
            subtalker_do_sample=request.subtalker_do_sample,
            subtalker_temperature=request.subtalker_temperature,
            subtalker_top_p=request.subtalker_top_p,
            subtalker_top_k=request.subtalker_top_k,
            truncated=take_audio.truncated,
            spoken_text=request.text,
            label=take_audio.label,
        )
        self.history.add(take)
        # Segments live only in memory; they back stem export for the latest take.
        if take_audio.segments:
            self._last_segments = (take.id, take_audio.segments, take.sample_rate)
        self.takes_panel.refresh()

        if self._capture_profile is not None:
            # This take is the golden sample; extract its speaker embedding as
            # soon as the job winds down (see _on_job_done).
            self._profile_pending = (str(wav_path), self._capture_profile)
            self._capture_profile = None

        if self._ab_capture is not None:
            self._ab_capture.append(take.id)
            if len(self._ab_capture) >= 2:
                self.takes_panel.slot_a = self._ab_capture[0]
                self.takes_panel.slot_b = self._ab_capture[1]
                self.takes_panel.refresh()
                self.status_label.setText(
                    "Pinned as A (with rule) and B (without) — Ctrl+1 / Ctrl+2"
                )
                self._ab_capture = None

        rtf = take.duration / take.elapsed if take.elapsed > 0 else 0.0
        self.perf_label.setText(f"{take.elapsed:.1f}s · {rtf:.2f}× realtime")
        if take_audio.truncated:
            # Otherwise a cut-off take is indistinguishable from a short one.
            self.status_label.setText(
                f"⚠ Take hit the {request.max_new_tokens}-frame ceiling and was cut "
                "off — raise Max tokens or shorten the text"
            )

        self.player.load(wav_path, take_audio.waveform, take.title)
        if self._autoplay_next:
            self.player.play()
            self._autoplay_next = False

    def _on_job_done(self, mode: str) -> None:
        self._busy = False
        self.script_panel.set_busy(False)
        self.progress.setVisible(False)
        self.status_label.setText(
            "Cancelled" if mode == "cancelled" else "Ready"
        )
        self._capture_profile = None
        if self._profile_pending is not None:
            wav_path, name = self._profile_pending
            self._profile_pending = None
            if mode != "cancelled":
                self._request_profile_extraction(wav_path, name)

    def _on_job_failed(self, message: str) -> None:
        self._busy = False
        self.script_panel.set_busy(False)
        self.progress.setVisible(False)
        self._capture_profile = None
        self._profile_pending = None
        self.status_label.setText("Generation failed")
        QMessageBox.warning(self, "Generation failed", message)

    # ---------- take actions ----------

    def _play_take(self, take_id: str) -> None:
        take = self.history.get(take_id)
        if take is None:
            return
        if not take.exists:
            QMessageBox.warning(self, "Missing audio", "That take's WAV file is gone.")
            return
        wave, _ = audio_utils.read_wav(take.wav_path)
        self.player.load(take.wav_path, wave, take.title)
        self.player.play()

    def _reroll(self, take_id: str) -> None:
        take = self.history.get(take_id)
        if take is None:
            return
        seed = new_seed()
        self.params_panel.show_seed(seed)
        request = SynthRequest(
            text=take.text,
            instruct=take.instruct,
            language=take.language,
            seed=seed,
            temperature=take.temperature,
            top_p=take.top_p,
            top_k=take.top_k,
            repetition_penalty=take.repetition_penalty,
            max_new_tokens=take.max_new_tokens,
            subtalker_do_sample=take.subtalker_do_sample,
            subtalker_temperature=take.subtalker_temperature,
            subtalker_top_p=take.subtalker_top_p,
            subtalker_top_k=take.subtalker_top_k,
        )
        if self.voice_profile is not None:
            # Rerolls follow the active profile so they stay in the same voice.
            request = replace(
                request,
                voice_profile=self.voice_profile,
                temperature=min(take.temperature, CLONE_TEMPERATURE),
            )
        self._submit(
            SynthJob(request=request, chunks=[take.text], seeds=[seed], mode=SINGLE),
            autoplay=True,
        )

    def _restore_params(self, take_id: str) -> None:
        take = self.history.get(take_id)
        if take is None:
            return
        self.design_panel.set_instruct(take.instruct, detached=True)
        self.script_panel.text_edit.setPlainText(take.text)
        if take.language:
            self.script_panel.language_combo.setCurrentText(take.language)
        self.params_panel.apply(
            {
                "temperature": take.temperature,
                "top_p": take.top_p,
                "top_k": take.top_k,
                "repetition_penalty": take.repetition_penalty,
                "max_new_tokens": take.max_new_tokens,
                "subtalker_do_sample": take.subtalker_do_sample,
                "subtalker_temperature": take.subtalker_temperature,
                "subtalker_top_p": take.subtalker_top_p,
                "subtalker_top_k": take.subtalker_top_k,
            }
        )
        self.params_panel.show_seed(take.seed)
        self.status_label.setText(f"Restored settings from take (seed {take.seed})")

    def _export_take(self, take_id: str) -> None:
        take = self.history.get(take_id)
        if take is None or not take.exists:
            return
        suggested = str(EXPORTS_DIR / f"{_safe_name(take.title)}_{take.seed}.wav")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export WAV", suggested, "WAV audio (*.wav)"
        )
        if not path:
            return
        try:
            shutil.copyfile(take.wav_path, path)
            self.status_label.setText(f"Exported to {Path(path).name}")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_stems_of_latest(self) -> None:
        """One WAV per dialogue line, plus a cue sheet of line timings."""
        if not self._last_segments:
            QMessageBox.information(
                self,
                "No dialogue to export",
                "Generate a dialogue script first — stems come from its lines.",
            )
            return

        take_id, segments, sample_rate = self._last_segments
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for stems", str(EXPORTS_DIR)
        )
        if not directory:
            return

        target = Path(directory)
        try:
            for index, segment in enumerate(segments, start=1):
                name = f"{index:04d}_{_safe_name(segment.speaker or 'line')}.wav"
                audio_utils.write_wav(target / name, segment.waveform, sample_rate)

            cue = dialogue_mod.format_cue_sheet(
                [
                    dialogue_mod.DialogueLine(speaker=s.speaker, text=s.text)
                    for s in segments
                ],
                [s.duration for s in segments],
                [s.gap_after for s in segments],
            )
            (target / "cue_sheet.tsv").write_text(cue)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        self.status_label.setText(
            f"Exported {len(segments)} stems and a cue sheet to {target.name}"
        )

    # ---------- settings ----------

    def _update_vram(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                used = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.device_label.setToolTip(f"VRAM reserved {used:.1f} / {total:.0f} GiB")
        except Exception:
            pass

    def _restore_settings(self) -> None:
        s = self.settings
        if s.instruct:
            self.design_panel.set_instruct(s.instruct, detached=True)
        self.script_panel.text_edit.setPlainText(s.script)
        self.script_panel.split_check.setChecked(s.split_long_script)
        self.script_panel.variations_spin.setValue(max(2, min(8, s.variations)))
        self.params_panel.set_linked(s.subtalker_linked)
        restored = {
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "repetition_penalty": s.repetition_penalty,
            "max_new_tokens": s.max_new_tokens,
        }
        if not s.subtalker_linked:
            restored.update(
                {
                    "subtalker_do_sample": s.subtalker_do_sample,
                    "subtalker_temperature": s.subtalker_temperature,
                    "subtalker_top_p": s.subtalker_top_p,
                    "subtalker_top_k": s.subtalker_top_k,
                }
            )
        self.params_panel.apply(restored)
        self.params_panel.seed_lock.setChecked(s.seed_locked)
        self.cast_panel.set_cast(s.cast)
        self.cast_panel.lock_voices.setChecked(s.lock_voices)
        if s.voice_profile:
            # The embedding was extracted once and saved; restoring it needs no
            # model at all, so a profile survives restarts for free.
            try:
                self.voice_profile = VoiceProfile.load(s.voice_profile)
                self._profile_path = Path(s.voice_profile)
                self.design_panel.set_profile_status(self.voice_profile.name)
            except Exception:
                s.voice_profile = ""
        if s.seed:
            self.params_panel.show_seed(s.seed)
        if len(s.window_geometry) == 4:
            # Saved under a different scale factor (or monitor), the stored
            # geometry can exceed the screen — clamp it, or the window comes
            # back taller than the work area and can't be managed sensibly.
            x, y, width, height = s.window_geometry
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                width = min(width, avail.width())
                height = min(height, avail.height())
                x = max(avail.left(), min(x, avail.right() - width + 1))
                y = max(avail.top(), min(y, avail.bottom() - height + 1))
            self.setGeometry(x, y, width, height)

    def closeEvent(self, event) -> None:
        s = self.settings
        s.instruct = self.design_panel.instruct
        s.script = self.script_panel.text_edit.toPlainText()
        s.language = self.script_panel.language
        s.split_long_script = self.script_panel.split_enabled
        s.variations = self.script_panel.variations_spin.value()
        s.seed = self.params_panel.seed_spin.value()
        s.seed_locked = self.params_panel.seed_locked
        s.subtalker_linked = self.params_panel.subtalker_link.isChecked()
        s.cast = self.cast_panel.cast
        s.lock_voices = self.cast_panel.lock_voices.isChecked()
        s.voice_profile = (
            str(self._profile_path)
            if self.voice_profile is not None and self._profile_path is not None
            else ""
        )
        s.__dict__.update(self.params_panel.values())
        geo = self.geometry()
        s.window_geometry = [geo.x(), geo.y(), geo.width(), geo.height()]
        try:
            s.save()
        except OSError:
            pass

        if self.pronounce_window is not None:
            self.pronounce_window.close()

        self.host.cancel()
        self.thread.quit()
        self.thread.wait(5000)
        super().closeEvent(event)


def _shrinkable(panel: QWidget) -> QScrollArea:
    """Wrap a column so its content minimum can't become the window minimum.

    Full width is preserved (content never scrolls sideways); vertically the
    column scrolls instead of forbidding the window from shrinking.
    """
    scroll = QScrollArea()
    scroll.setWidget(panel)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setMinimumWidth(
        panel.minimumSizeHint().width()
        + scroll.verticalScrollBar().sizeHint().width()
    )
    return scroll


def _safe_name(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_ " else "_" for c in text[:40]]
    return "".join(keep).strip().replace(" ", "_") or "take"


def _profile_name(instruct: str) -> str:
    """A short display name for a profile extracted from a description."""
    words = instruct.split()
    if not words:
        return "Voice profile"
    name = " ".join(words[:5])
    return name + "…" if len(words) > 5 else name
