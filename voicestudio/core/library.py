"""User voice preset library — forks of templates plus hand-written voices."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from .config import PRESETS_FILE, ensure_dirs


@dataclass
class VoicePreset:
    """A saved voice.

    Measured behaviour (see scripts/seed_study.py): the *description* carries
    speaker identity — the same description at different seeds produces the same
    voice (timbre similarity 0.93, F0 within ~3 Hz), while a different
    description moves it enormously (0.59, ~156 Hz).

    So `seed` is not what holds the voice together across lines. What it pins is
    an exact rendition: identical text plus identical seed reproduces bit-identical
    audio. Store it to get *that take* back; the voice itself survives without it.
    """

    name: str
    instruct: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tags: list[str] = field(default_factory=list)
    traits: dict = field(default_factory=dict)
    favorite: bool = False
    source_template: str | None = None
    seed: int | None = None
    created: float = field(default_factory=time.time)

    @property
    def seed_pinned(self) -> bool:
        return self.seed is not None

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        haystack = f"{self.name} {self.instruct} {' '.join(self.tags)}".lower()
        return all(term in haystack for term in q.split())


class VoiceLibrary:
    """JSON-backed preset store. Saves on every mutation."""

    def __init__(self, presets: list[VoicePreset] | None = None):
        self._presets: list[VoicePreset] = presets or []

    @classmethod
    def load(cls) -> "VoiceLibrary":
        try:
            raw = json.loads(PRESETS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = set(VoicePreset.__dataclass_fields__)
        items = [
            VoicePreset(**{k: v for k, v in entry.items() if k in known})
            for entry in raw.get("presets", [])
        ]
        return cls(items)

    def save(self) -> None:
        ensure_dirs()
        payload = {"version": 1, "presets": [asdict(p) for p in self._presets]}
        tmp = PRESETS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(PRESETS_FILE)

    def all(self, query: str = "") -> list[VoicePreset]:
        """Favorites first, then newest."""
        items = [p for p in self._presets if p.matches(query)]
        return sorted(items, key=lambda p: (not p.favorite, -p.created))

    def get(self, preset_id: str) -> VoicePreset | None:
        return next((p for p in self._presets if p.id == preset_id), None)

    def add(self, preset: VoicePreset) -> VoicePreset:
        preset.name = self._unique_name(preset.name)
        self._presets.append(preset)
        self.save()
        return preset

    def update(self, preset: VoicePreset) -> None:
        for i, existing in enumerate(self._presets):
            if existing.id == preset.id:
                self._presets[i] = preset
                self.save()
                return

    def remove(self, preset_id: str) -> None:
        self._presets = [p for p in self._presets if p.id != preset_id]
        self.save()

    def duplicate(self, preset_id: str) -> VoicePreset | None:
        src = self.get(preset_id)
        if src is None:
            return None
        copy = VoicePreset(
            name=src.name,
            instruct=src.instruct,
            tags=list(src.tags),
            traits=dict(src.traits),
            source_template=src.source_template,
            seed=src.seed,
        )
        return self.add(copy)

    def toggle_favorite(self, preset_id: str) -> None:
        p = self.get(preset_id)
        if p is not None:
            p.favorite = not p.favorite
            self.save()

    def _unique_name(self, name: str) -> str:
        name = (name or "Untitled voice").strip()
        existing = {p.name for p in self._presets}
        if name not in existing:
            return name
        n = 2
        while f"{name} ({n})" in existing:
            n += 1
        return f"{name} ({n})"
