"""Waveform view and transport bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core import audio as audio_utils
from . import theme
from .metrics import metrics


class WaveformView(QWidget):
    """Peak-envelope waveform with a playhead. Click or drag to seek."""

    seekRequested = pyqtSignal(float)  # 0..1

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._envelope = np.zeros(0, dtype=np.float32)
        self._wave: np.ndarray | None = None
        self._progress = 0.0
        m = metrics()
        self.setMinimumHeight(m.sp(5.5))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        m = metrics()
        return QSize(m.ch(60), m.sp(6.5))

    def set_waveform(self, wave: np.ndarray | None) -> None:
        self._wave = wave
        self._progress = 0.0
        self._rebuild()
        self.update()

    def set_progress(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        if abs(fraction - self._progress) > 0.001:
            self._progress = fraction
            self.update()

    def clear(self) -> None:
        self.set_waveform(None)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild()

    def _bucket_count(self) -> int:
        return max(24, self.width() // 2)

    def _rebuild(self) -> None:
        if self._wave is None or self._wave.size == 0:
            self._envelope = np.zeros(0, dtype=np.float32)
        else:
            self._envelope = audio_utils.envelope(self._wave, self._bucket_count())

    def mousePressEvent(self, event) -> None:
        self._seek_to(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def _seek_to(self, x: float) -> None:
        if self._envelope.size and self.width() > 0:
            self.seekRequested.emit(min(1.0, max(0.0, x / self.width())))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        rect = self.rect()
        painter.fillRect(rect, theme.WAVE_BG)

        if self._envelope.size == 0:
            painter.setPen(QPen(QColor(theme.TEXT_DIM)))
            painter.drawText(
                rect, Qt.AlignmentFlag.AlignCenter, "No audio — generate a take"
            )
            painter.end()
            return

        mid = rect.height() / 2.0
        n = self._envelope.size
        step = rect.width() / n
        played_until = self._progress * rect.width()

        # Zero line
        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawLine(0, int(mid), rect.width(), int(mid))

        for i, value in enumerate(self._envelope):
            x = i * step
            half = max(1.0, float(value) * (mid - 4))
            color = theme.WAVE_PLAYED if x <= played_until else theme.WAVE
            painter.setPen(QPen(color, max(1.0, step * 0.8)))
            painter.drawLine(
                int(x + step / 2), int(mid - half), int(x + step / 2), int(mid + half)
            )

        if self._progress > 0:
            painter.setPen(QPen(theme.PLAYHEAD, 1))
            painter.drawLine(int(played_until), 0, int(played_until), rect.height())

        painter.end()


class PlayerBar(QWidget):
    """Transport controls plus the waveform for the currently loaded take."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = 0
        self._path: Path | None = None

        self.player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self._audio_out)
        self._audio_out.setVolume(0.9)

        self.waveform = WaveformView(self)
        self.waveform.seekRequested.connect(self._seek_fraction)

        m = metrics()
        self.play_button = QPushButton("▶")
        self.play_button.setMinimumWidth(m.ch(4))
        self.play_button.setToolTip("Play / pause (Space)")
        self.play_button.clicked.connect(self.toggle)
        self.play_button.setEnabled(False)

        self.stop_button = QPushButton("■")
        self.stop_button.setMinimumWidth(m.icon_button())
        self.stop_button.setProperty("role", "icon")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        self.title_label = QLabel("—")
        self.title_label.setProperty("role", "hint")
        self.title_label.setWordWrap(False)

        self.time_label = QLabel("0.0s / 0.0s")
        self.time_label.setProperty("role", "metric")

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(m.sp(0.35))
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.title_label, 1)
        controls.addWidget(self.time_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(0.35))
        layout.addWidget(self.waveform)
        layout.addLayout(controls)

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)

    def load(self, path: Path | str, wave: np.ndarray | None, title: str = "") -> None:
        self._path = Path(path)
        self.waveform.set_waveform(wave)
        self.title_label.setText(title or self._path.name)
        self.player.setSource(QUrl.fromLocalFile(str(self._path)))
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def play(self) -> None:
        if self._path is not None:
            self.player.play()

    def toggle(self) -> None:
        if self._path is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop(self) -> None:
        self.player.stop()
        self.waveform.set_progress(0.0)

    def clear(self) -> None:
        self.stop()
        self._path = None
        self.player.setSource(QUrl())
        self.waveform.clear()
        self.title_label.setText("—")
        self.time_label.setText("0.0s / 0.0s")
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def _seek_fraction(self, fraction: float) -> None:
        if self._duration_ms > 0:
            self.player.setPosition(int(fraction * self._duration_ms))

    def _on_position(self, ms: int) -> None:
        if self._duration_ms > 0:
            self.waveform.set_progress(ms / self._duration_ms)
        self.time_label.setText(
            f"{ms / 1000:.1f}s / {self._duration_ms / 1000:.1f}s"
        )

    def _on_duration(self, ms: int) -> None:
        self._duration_ms = max(0, ms)
        self.time_label.setText(f"0.0s / {self._duration_ms / 1000:.1f}s")

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("❚❚" if playing else "▶")
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.waveform.set_progress(0.0)
