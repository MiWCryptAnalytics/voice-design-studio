"""Application paths and persisted settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _data_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "voicestudio"


DATA_DIR = _data_root()
TAKES_DIR = DATA_DIR / "takes"
EXPORTS_DIR = DATA_DIR / "exports"
PROFILES_DIR = DATA_DIR / "profiles"
PRESETS_FILE = DATA_DIR / "presets.json"
HISTORY_FILE = DATA_DIR / "history.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
TEMPLATES_FILE = ASSETS_DIR / "templates.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, TAKES_DIR, EXPORTS_DIR, PROFILES_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Session state restored on launch."""

    instruct: str = ""
    script: str = "The quiet between two thoughts is where the whole story hides."
    language: str = "English"
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 8192
    subtalker_linked: bool = True
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50
    seed: int = 0
    seed_locked: bool = False
    variations: int = 4
    split_long_script: bool = True
    cast: dict = field(default_factory=dict)  # speaker -> voice key
    lock_voices: bool = True
    window_geometry: list[int] = field(default_factory=list)
    # Path to the active VoiceProfile .npz; empty means text-description mode.
    voice_profile: str = ""

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(SETTINGS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        ensure_dirs()
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(SETTINGS_FILE)
