"""Background model host.

The model is loaded and driven entirely inside a worker QThread. The UI talks to
it only through queued signals, so neither the cold load nor a multi-second
generation ever blocks painting.

A job is a list of `JobItem`s plus a mode that says how their audio combines:
one take, one joined take, or a take per item.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field, replace

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ..core.segment import split_on_silence
from ..engine import SynthEngine, SynthRequest, load_model, new_seed

# Joining a speaker's lines with a plain space is what produces a clean silence
# at each sentence boundary; decorated joiners ("…") produce extra gaps that
# defeat the split.
LINE_JOINER = " "
# Keep a single pass to a sensible length so it stays coherent and splittable.
MAX_GROUP_CHARS = 400

# Job modes
SINGLE = "single"          # one item -> one take
SCRIPT = "script"          # many items -> one joined take
VARIATIONS = "variations"  # one item, many seeds -> a take each
COMPARE = "compare"        # many items, one seed -> a take each
DIALOGUE = "dialogue"      # many items with own voices -> one joined take + stems

_JOINED = (SCRIPT, DIALOGUE)
_PER_ITEM = (COMPARE,)


@dataclass
class JobItem:
    """One utterance. `instruct` overrides the job's voice, for dialogue."""

    text: str
    instruct: str | None = None
    label: str = ""
    speaker: str = ""
    pause_after: float | None = None


@dataclass
class Segment:
    """A rendered item inside a joined take — used for stems and cue sheets."""

    speaker: str
    text: str
    waveform: np.ndarray
    duration: float
    gap_after: float


@dataclass
class SynthJob:
    request: SynthRequest
    items: list[JobItem] = field(default_factory=list)
    chunks: list[str] | None = None  # convenience: plain text items
    seeds: list[int] = field(default_factory=list)
    mode: str = SINGLE
    label: str = ""
    gap_seconds: float = 0.25
    # Dialogue only: render each speaker's lines in one pass and split them, which
    # is the only mechanism this checkpoint offers for holding a voice steady.
    lock_voices: bool = True

    def __post_init__(self) -> None:
        if self.chunks and not self.items:
            self.items = [JobItem(text=t) for t in self.chunks]
        if not self.seeds:
            base = self.request.seed if self.request.seed is not None else new_seed()
            self.seeds = [base]

    @property
    def total_steps(self) -> int:
        return len(self.seeds) * max(1, len(self.items))


@dataclass
class TakeAudio:
    """A finished take, ready for the UI to store and play."""

    waveform: np.ndarray
    sample_rate: int
    seed: int
    elapsed: float
    request: SynthRequest
    label: str = ""
    truncated: bool = False
    segments: list[Segment] = field(default_factory=list)

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
    jobDone = pyqtSignal(str)  # mode, or "cancelled"
    jobFailed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._engine: SynthEngine | None = None
        self._cancel = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    def cancel(self) -> None:
        """Thread-safe; checked between items and between seeds."""
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
        if not job.items:
            self.jobFailed.emit("Nothing to generate.")
            return

        self._cancel.clear()
        self.jobStarted.emit(job.total_steps)
        done = 0

        try:
            for index, seed in enumerate(job.seeds):
                if self._cancel.is_set():
                    break
                request = replace(job.request, seed=seed)

                if job.mode == DIALOGUE and job.lock_voices and len(job.items) > 1:
                    take = self._run_voice_locked(job, request, done)
                    done += len(job.items)
                    if take is None:
                        break
                    self.takeReady.emit(take)

                elif job.mode in _JOINED and len(job.items) > 1:
                    take = self._run_joined(job, request, done)
                    done += len(job.items)
                    if take is None:
                        break
                    self.takeReady.emit(take)

                elif job.mode in _PER_ITEM:
                    for item in job.items:
                        if self._cancel.is_set():
                            break
                        take = self._run_one(job, request, item, item.label, done)
                        done += 1
                        self.takeReady.emit(take)
                        self.jobProgress.emit(item.label or "Generating…", done,
                                              job.total_steps)
                    if self._cancel.is_set():
                        break

                else:
                    item = job.items[0]
                    label = job.label or item.label
                    if job.mode == VARIATIONS:
                        label = f"Variation {index + 1} of {len(job.seeds)}"
                    take = self._run_one(job, request, item, label, done)
                    done += 1
                    self.takeReady.emit(take)
                    self.jobProgress.emit(label or "Generating…", done, job.total_steps)

            self.jobDone.emit("cancelled" if self._cancel.is_set() else job.mode)
        except Exception as exc:
            traceback.print_exc()
            self.jobFailed.emit(f"{type(exc).__name__}: {exc}")

    # ---------- generation shapes ----------

    def _synth(self, request: SynthRequest, item: JobItem):
        per_item = replace(request, text=item.text)
        if item.instruct is not None:
            per_item = replace(per_item, instruct=item.instruct)
        return self._engine.synthesize(per_item)

    def _run_one(
        self, job: SynthJob, request: SynthRequest, item: JobItem,
        label: str, done_before: int,
    ) -> TakeAudio:
        self.jobProgress.emit(label or "Generating…", done_before, job.total_steps)
        result = self._synth(request, item)
        return TakeAudio(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            seed=result.seed,
            elapsed=result.elapsed,
            request=result.request,
            label=label,
            truncated=result.truncated,
        )

    def _run_voice_locked(
        self, job: SynthJob, request: SynthRequest, done_before: int
    ) -> TakeAudio | None:
        """Render each speaker's whole part in one pass, then cut it into lines.

        A voice only stays genuinely fixed inside a single generation, so every
        line a character speaks is produced together and split afterwards. If a
        split can't be made safely the group falls back to line-by-line, which
        costs consistency but never ships mis-cut audio.
        """
        groups = _group_by_speaker(job.items)
        rendered: dict[int, np.ndarray] = {}
        elapsed = 0.0
        truncated = False
        sample_rate = self._engine.sample_rate
        done = done_before
        fallbacks = 0

        for speaker, indexed in groups:
            for batch in _batches(indexed, MAX_GROUP_CHARS):
                if self._cancel.is_set():
                    return None
                positions = [i for i, _ in batch]
                items = [item for _, item in batch]
                self.jobProgress.emit(
                    f"{speaker}: {len(items)} line{'s' if len(items) > 1 else ''}",
                    done, job.total_steps,
                )

                segments = None
                if len(items) > 1:
                    merged = JobItem(
                        text=LINE_JOINER.join(i.text for i in items),
                        instruct=items[0].instruct,
                        speaker=speaker,
                    )
                    result = self._synth(request, merged)
                    elapsed += result.elapsed
                    truncated = truncated or result.truncated
                    sample_rate = result.sample_rate
                    segments = split_on_silence(
                        result.waveform, sample_rate, [i.text for i in items]
                    )
                    if segments is None:
                        fallbacks += 1

                if segments is None:
                    for position, item in zip(positions, items):
                        if self._cancel.is_set():
                            return None
                        result = self._synth(request, item)
                        rendered[position] = result.waveform
                        elapsed += result.elapsed
                        truncated = truncated or result.truncated
                        sample_rate = result.sample_rate
                else:
                    for position, wave in zip(positions, segments):
                        rendered[position] = wave

                done += len(items)

        if not rendered:
            return None

        segments_out = [
            Segment(
                speaker=item.speaker,
                text=item.text,
                waveform=rendered[index],
                duration=len(rendered[index]) / float(sample_rate or 1),
                gap_after=(
                    (job.gap_seconds if item.pause_after is None else item.pause_after)
                    if index < len(job.items) - 1 else 0.0
                ),
            )
            for index, item in enumerate(job.items)
            if index in rendered
        ]

        label = job.label or f"Dialogue · {len(segments_out)} lines"
        if fallbacks:
            label += f" ({fallbacks} group{'s' if fallbacks > 1 else ''} unsplit)"

        return TakeAudio(
            waveform=_concat_with_gaps(segments_out, sample_rate),
            sample_rate=sample_rate,
            seed=request.seed,
            elapsed=elapsed,
            request=replace(request, text=" ".join(s.text for s in segments_out)),
            label=label,
            truncated=truncated,
            segments=segments_out,
        )

    def _run_joined(
        self, job: SynthJob, request: SynthRequest, done_before: int
    ) -> TakeAudio | None:
        """Render every item and concatenate.

        Continuity across items comes from the shared *description* (per item for
        dialogue), not the seed — measurement shows seed choice barely moves
        speaker identity when the text differs.
        """
        segments: list[Segment] = []
        elapsed = 0.0
        truncated = False
        sample_rate = self._engine.sample_rate
        total = len(job.items)

        for i, item in enumerate(job.items):
            if self._cancel.is_set():
                return None
            who = f"{item.speaker}: " if item.speaker else ""
            self.jobProgress.emit(
                f"{who}{i + 1} of {total}", done_before + i, job.total_steps
            )
            result = self._synth(request, item)
            gap = job.gap_seconds if item.pause_after is None else item.pause_after
            segments.append(
                Segment(
                    speaker=item.speaker,
                    text=item.text,
                    waveform=result.waveform,
                    duration=result.duration,
                    gap_after=gap if i < total - 1 else 0.0,
                )
            )
            elapsed += result.elapsed
            truncated = truncated or result.truncated
            sample_rate = result.sample_rate

        if not segments:
            return None

        joined = _concat_with_gaps(segments, sample_rate)
        default_label = (
            f"Dialogue · {total} lines" if job.mode == DIALOGUE
            else f"Script · {total} chunks"
        )
        return TakeAudio(
            waveform=joined,
            sample_rate=sample_rate,
            seed=request.seed,
            elapsed=elapsed,
            request=replace(request, text=" ".join(s.text for s in segments)),
            label=job.label or default_label,
            truncated=truncated,
            segments=segments,
        )


def _group_by_speaker(items: list[JobItem]) -> list[tuple[str, list[tuple[int, JobItem]]]]:
    """All of a speaker's lines together, keeping their original positions.

    Order follows first appearance so progress reads in a sensible order.
    """
    groups: dict[str, list[tuple[int, JobItem]]] = {}
    order: list[str] = []
    for index, item in enumerate(items):
        key = item.speaker or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, item))
    return [(speaker, groups[speaker]) for speaker in order]


def _batches(
    indexed: list[tuple[int, JobItem]], max_chars: int
) -> list[list[tuple[int, JobItem]]]:
    """Split one speaker's lines into passes short enough to stay coherent."""
    batches: list[list[tuple[int, JobItem]]] = []
    current: list[tuple[int, JobItem]] = []
    size = 0
    for entry in indexed:
        length = len(entry[1].text)
        if current and size + length > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(entry)
        size += length
    if current:
        batches.append(current)
    return batches


def _concat_with_gaps(segments: list[Segment], sample_rate: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for segment in segments:
        pieces.append(segment.waveform.astype(np.float32))
        if segment.gap_after > 0:
            pieces.append(
                np.zeros(int(segment.gap_after * sample_rate), dtype=np.float32)
            )
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
