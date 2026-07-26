"""Take history — every generation, newest first, with its params and audio."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import HISTORY_FILE, TAKES_DIR, ensure_dirs

MAX_TAKES = 300


@dataclass
class Take:
    instruct: str
    text: str
    language: str
    seed: int
    wav_path: str
    sample_rate: int
    duration: float
    elapsed: float
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 4096
    starred: bool = False
    label: str = ""
    created: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def exists(self) -> bool:
        return Path(self.wav_path).is_file()

    @property
    def title(self) -> str:
        if self.label:
            return self.label
        head = " ".join(self.instruct.split())
        return (head[:60] + "…") if len(head) > 60 else (head or "(no description)")

    @property
    def when(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.created))


class TakeHistory:
    """JSON-backed log of generations; audio lives beside it as WAV files."""

    def __init__(self, takes: list[Take] | None = None):
        self._takes: list[Take] = takes or []

    @classmethod
    def load(cls) -> "TakeHistory":
        try:
            raw = json.loads(HISTORY_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = set(Take.__dataclass_fields__)
        items = [
            Take(**{k: v for k, v in entry.items() if k in known})
            for entry in raw.get("takes", [])
        ]
        # Drop entries whose audio was deleted outside the app.
        return cls([t for t in items if t.exists])

    def save(self) -> None:
        ensure_dirs()
        payload = {"version": 1, "takes": [asdict(t) for t in self._takes]}
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(HISTORY_FILE)

    def all(self) -> list[Take]:
        return list(self._takes)

    def get(self, take_id: str) -> Take | None:
        return next((t for t in self._takes if t.id == take_id), None)

    def add(self, take: Take) -> Take:
        self._takes.insert(0, take)
        self._prune()
        self.save()
        return take

    def remove(self, take_id: str) -> None:
        take = self.get(take_id)
        if take is None:
            return
        self._takes = [t for t in self._takes if t.id != take_id]
        _unlink(take.wav_path)
        self.save()

    def toggle_star(self, take_id: str) -> None:
        take = self.get(take_id)
        if take is not None:
            take.starred = not take.starred
            self.save()

    def clear_unstarred(self) -> None:
        keep, drop = [], []
        for t in self._takes:
            (keep if t.starred else drop).append(t)
        for t in drop:
            _unlink(t.wav_path)
        self._takes = keep
        self.save()

    def next_wav_path(self) -> Path:
        ensure_dirs()
        return TAKES_DIR / f"take_{uuid.uuid4().hex[:12]}.wav"

    def _prune(self) -> None:
        """Cap history size, discarding oldest unstarred takes first."""
        if len(self._takes) <= MAX_TAKES:
            return
        overflow = len(self._takes) - MAX_TAKES
        for take in reversed(self._takes):
            if overflow <= 0:
                break
            if not take.starred:
                self._takes.remove(take)
                _unlink(take.wav_path)
                overflow -= 1


def _unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
