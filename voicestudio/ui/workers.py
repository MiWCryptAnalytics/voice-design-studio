"""Background model host.

The model is loaded and driven entirely inside a worker QThread. The UI talks to
it only through queued signals, so neither the ~30s cold load nor a multi-second
generation ever blocks painting.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field, replace

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ..core import audio as audio_utils
from ..engine import SynthEngine, SynthRequest, load_model, new_seed

# Job modes
SINGLE = "single"
VARIATIONS = "variations"
SCRIPT = "script"


@dataclass
class SynthJob:
    """One unit of work queued onto the engine thread."""

    request: SynthRequest
    chunks: list[str]
    seeds: list[int] = field(default_factory=list)
    mode: str = SINGLE
    label: str = ""
    gap_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not self.seeds:
            base = self.request.seed if self.request.seed is not None else new_seed()
            self.seeds = [base]

    @property
    def total_steps(self) -> int:
        return len(self.seeds) * max(1, len(self.chunks))


@dataclass
class TakeAudio:
    """A finished take, ready for the UI to store and play."""

    waveform: np.ndarray
    sample_rate: int
    seed: int
    elapsed: float
    request: SynthRequest
    label: str = ""

    @property
    def duration(self) -> float:
        return len(self.waveform) / float(self.sample_rate or 1)


class EngineHost(QObject):
    """Lives on the engine thread; owns the model."""

    loadProgress = pyqtSignal(str)
    loadReady = pyqtSignal(object)  # LoadedModel
    loadFailed = pyqtSignal(str)

    jobStarted = pyqtSignal(int)  # total steps
    jobProgress = pyqtSignal(str, int, int)  # message, done, total
    takeReady = pyqtSignal(object)  # TakeAudio
    jobDone = pyqtSignal(str)  # mode
    jobFailed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._engine: SynthEngine | None = None
        self._cancel = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    def cancel(self) -> None:
        """Thread-safe; checked between chunks and variations."""
        self._cancel.set()

    @pyqtSlot()
    def load(self) -> None:
        try:
            loaded = load_model(progress=self.loadProgress.emit)
            self._engine = SynthEngine(loaded)
            self.loadReady.emit(loaded)
        except Exception as exc:
            traceback.print_exc()
            self.loadFailed.emit(f"{type(exc).__name__}: {exc}")

    @pyqtSlot(object)
    def run_job(self, job: SynthJob) -> None:
        if self._engine is None:
            self.jobFailed.emit("Model is not loaded yet.")
            return

        self._cancel.clear()
        self.jobStarted.emit(job.total_steps)
        done = 0

        try:
            for index, seed in enumerate(job.seeds):
                if self._cancel.is_set():
                    break

                request = replace(job.request, seed=seed)
                label = job.label
                if job.mode == VARIATIONS:
                    label = f"Variation {index + 1} of {len(job.seeds)}"

                if job.mode == SCRIPT and len(job.chunks) > 1:
                    take = self._run_script(job, request, label, done)
                    done += len(job.chunks)
                else:
                    self.jobProgress.emit(
                        label or "Generating…", done, job.total_steps
                    )
                    result = self._engine.synthesize(
                        replace(request, text=job.chunks[0])
                    )
                    done += 1
                    take = TakeAudio(
                        waveform=result.waveform,
                        sample_rate=result.sample_rate,
                        seed=result.seed,
                        elapsed=result.elapsed,
                        request=result.request,
                        label=label,
                    )

                if take is None:  # cancelled mid-script
                    break

                self.takeReady.emit(take)
                self.jobProgress.emit(label or "Generating…", done, job.total_steps)

            self.jobDone.emit("cancelled" if self._cancel.is_set() else job.mode)
        except Exception as exc:
            traceback.print_exc()
            self.jobFailed.emit(f"{type(exc).__name__}: {exc}")

    def _run_script(
        self, job: SynthJob, request: SynthRequest, label: str, done_before: int
    ) -> TakeAudio | None:
        """Generate each chunk under one description and seed, then concatenate."""
        pieces: list[np.ndarray] = []
        elapsed = 0.0
        sample_rate = self._engine.sample_rate

        for i, chunk in enumerate(job.chunks):
            if self._cancel.is_set():
                return None
            self.jobProgress.emit(
                f"Chunk {i + 1} of {len(job.chunks)}",
                done_before + i,
                job.total_steps,
            )
            result = self._engine.synthesize(replace(request, text=chunk))
            pieces.append(result.waveform)
            elapsed += result.elapsed
            sample_rate = result.sample_rate

        if not pieces:
            return None

        return TakeAudio(
            waveform=audio_utils.concat(pieces, sample_rate, job.gap_seconds),
            sample_rate=sample_rate,
            seed=request.seed,
            elapsed=elapsed,
            request=replace(request, text=" ".join(job.chunks)),
            label=label or f"Script · {len(job.chunks)} chunks",
        )
