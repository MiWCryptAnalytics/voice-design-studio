"""Main studio window: layout, engine thread wiring, and the generate loop."""

from __future__ import annotations

import shutil
from pathlib import Path

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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..core import audio as audio_utils
from ..core.config import EXPORTS_DIR, Settings, ensure_dirs
from ..core.history import Take, TakeHistory
from ..core.library import VoiceLibrary
from ..core.templates import load_templates
from ..engine import SynthRequest, new_seed
from .design_panel import DesignPanel
from .metrics import metrics
from .params_panel import ParamsPanel
from .player import PlayerBar
from .script_panel import ScriptPanel
from .takes_panel import TakesPanel
from .workers import SCRIPT, SINGLE, VARIATIONS, EngineHost, SynthJob, TakeAudio


class MainWindow(QMainWindow):
    loadRequested = pyqtSignal()
    jobRequested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()

        self.settings = Settings.load()
        self.library = VoiceLibrary.load()
        self.history = TakeHistory.load()
        self.templates = load_templates()

        self._busy = False
        self._autoplay_next = False
        self._pending_takes = 0

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

        self.script_panel = ScriptPanel()
        self.script_panel.generateRequested.connect(self.generate_one)
        self.script_panel.variationsRequested.connect(self.generate_variations)
        self.script_panel.cancelRequested.connect(self._cancel_job)

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
        player_layout.setSpacing(m.sp(0.45))
        player_title = QLabel("PREVIEW")
        player_title.setProperty("role", "title")
        player_layout.addWidget(player_title)
        player_layout.addWidget(self.player)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(m.sp(0.6))
        center_layout.addWidget(self.script_panel, 3)
        center_layout.addWidget(self.params_panel, 0)
        center_layout.addWidget(player_frame, 0)

        self.takes_panel = TakesPanel(self.history)
        self.takes_panel.playRequested.connect(self._play_take)
        self.takes_panel.rerollRequested.connect(self._reroll)
        self.takes_panel.restoreRequested.connect(self._restore_params)
        self.takes_panel.exportRequested.connect(self._export_take)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.design_panel)
        splitter.addWidget(center)
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

        self._build_status_bar()

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

        self.host.loadProgress.connect(self._on_load_progress)
        self.host.loadReady.connect(self._on_load_ready)
        self.host.loadFailed.connect(self._on_load_failed)
        self.host.jobStarted.connect(self._on_job_started)
        self.host.jobProgress.connect(self._on_job_progress)
        self.host.takeReady.connect(self._on_take_ready)
        self.host.jobDone.connect(self._on_job_done)
        self.host.jobFailed.connect(self._on_job_failed)

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
        self.status_label.setText("Ready")
        self.device_label.setText(f"{loaded.device_label} · {loaded.sample_rate} Hz")
        self.script_panel.set_busy(False)

    def _on_load_failed(self, message: str) -> None:
        self.status_label.setText("Model failed to load")
        QMessageBox.critical(
            self,
            "Could not load model",
            f"{message}\n\nThe app will stay open, but generation is unavailable.",
        )

    # ---------- generating ----------

    def _build_request(self, text: str, seed: int) -> SynthRequest:
        params = self.params_panel.values()
        return SynthRequest(
            text=text,
            instruct=self.design_panel.instruct,
            language=self.script_panel.language,
            seed=seed,
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
        chunks = self.script_panel.chunks()
        if not chunks:
            self.status_label.setText("Write something for the voice to say.")
            return
        seed = self.params_panel.effective_seed()
        request = self._build_request(chunks[0], seed)
        mode = SCRIPT if len(chunks) > 1 else SINGLE
        self._submit(
            SynthJob(request=request, chunks=chunks, seeds=[seed], mode=mode),
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
                chunks=[chunks[0]],
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
                chunks=[self.templates.demo_sentence],
                seeds=[self.templates.demo_seed],
                mode=SINGLE,
                label=f"Audition · {name}",
            ),
            autoplay=True,
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
            label=take_audio.label,
        )
        self.history.add(take)
        self.takes_panel.refresh()

        rtf = take.duration / take.elapsed if take.elapsed > 0 else 0.0
        self.perf_label.setText(f"{take.elapsed:.1f}s · {rtf:.2f}× realtime")

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

    def _on_job_failed(self, message: str) -> None:
        self._busy = False
        self.script_panel.set_busy(False)
        self.progress.setVisible(False)
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
        self.params_panel.apply(
            {
                "temperature": s.temperature,
                "top_p": s.top_p,
                "top_k": s.top_k,
                "repetition_penalty": s.repetition_penalty,
                "max_new_tokens": s.max_new_tokens,
            }
        )
        self.params_panel.seed_lock.setChecked(s.seed_locked)
        if s.seed:
            self.params_panel.show_seed(s.seed)
        if len(s.window_geometry) == 4:
            self.setGeometry(*s.window_geometry)

    def closeEvent(self, event) -> None:
        s = self.settings
        s.instruct = self.design_panel.instruct
        s.script = self.script_panel.text_edit.toPlainText()
        s.language = self.script_panel.language
        s.split_long_script = self.script_panel.split_enabled
        s.variations = self.script_panel.variations_spin.value()
        s.seed = self.params_panel.seed_spin.value()
        s.seed_locked = self.params_panel.seed_locked
        s.__dict__.update(self.params_panel.values())
        geo = self.geometry()
        s.window_geometry = [geo.x(), geo.y(), geo.width(), geo.height()]
        try:
            s.save()
        except OSError:
            pass

        self.host.cancel()
        self.thread.quit()
        self.thread.wait(5000)
        super().closeEvent(event)


def _safe_name(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_ " else "_" for c in text[:40]]
    return "".join(keep).strip().replace(" ", "_") or "take"
