"""A fixed speaker identity: one embedding, extracted once, reused for every line.

The VoiceDesign checkpoint re-derives the speaker from the description text on
every generation, which is what lets a voice drift across a script. A profile
pins the speaker instead: the Base checkpoint's speaker encoder turns a
reference recording into a single x-vector, and generation conditions on that
same vector for every chunk.

This module is pure data (numpy only) so profiles can be saved, loaded and
tested without importing torch or loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VoiceProfile:
    """One speaker embedding plus enough provenance to name it in the UI."""

    name: str
    embedding: np.ndarray = field(repr=False)  # (D,) float32
    source_wav: str = ""

    def __post_init__(self) -> None:
        self.embedding = np.ascontiguousarray(
            np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        )
        if self.embedding.size == 0:
            raise ValueError("A voice profile needs a non-empty embedding.")

    def save(self, path: Path | str) -> Path:
        """Write the profile as a single .npz so it survives restarts."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embedding=self.embedding,
            name=np.array(self.name),
            source_wav=np.array(self.source_wav),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "VoiceProfile":
        data = np.load(Path(path), allow_pickle=False)
        return cls(
            name=str(data["name"]),
            embedding=data["embedding"],
            source_wav=str(data["source_wav"]),
        )
