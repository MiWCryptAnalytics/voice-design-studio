"""Split one generated utterance back into its constituent lines.

Why this exists: the VoiceDesign checkpoint has no speaker-locking mechanism —
`extract_speaker_embedding` needs a speaker encoder this checkpoint doesn't ship,
so ICL and voice cloning are unavailable. The only thing that holds a voice
genuinely steady is generating inside a single pass, where the audio is one
continuous sample.

So for dialogue we generate all of a character's lines at once and cut them apart
here. Measured on three voices, the model leaves a clear silence at each sentence
boundary and the resulting split matches the expected text proportions to within
a few percent.
"""

from __future__ import annotations

import numpy as np

FLOOR_DB = -38.0
MIN_SILENCE_MS = 110.0
WINDOW_S = 0.02
EDGE_GUARD_S = 0.25
# A split is rejected if any line's share of the audio is this far from the share
# its text length predicts — a cheap guard against cutting in the wrong place.
MAX_SHARE_ERROR = 0.16


def find_silences(
    wav: np.ndarray,
    sample_rate: int,
    floor_db: float = FLOOR_DB,
    min_ms: float = MIN_SILENCE_MS,
) -> list[tuple[float, float]]:
    """Interior quiet spans as (centre_sample, duration_ms), longest first."""
    window = int(sample_rate * WINDOW_S)
    if window <= 0 or wav.size < window * 2:
        return []

    count = wav.size // window
    frames = wav[: count * window].reshape(count, window)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-10
    peak = float(np.abs(wav).max()) + 1e-10
    db = 20 * np.log10(rms / peak)
    quiet = db < floor_db

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(quiet)))

    guard = EDGE_GUARD_S * sample_rate
    out: list[tuple[float, float]] = []
    for a, b in spans:
        duration_ms = (b - a) * window / sample_rate * 1000
        centre = (a + b) / 2 * window
        if duration_ms >= min_ms and guard < centre < wav.size - guard:
            out.append((centre, duration_ms))

    out.sort(key=lambda s: -s[1])
    return out


def split_on_silence(
    wav: np.ndarray,
    sample_rate: int,
    texts: list[str],
) -> list[np.ndarray] | None:
    """Cut `wav` into one segment per text, or None if it can't be done safely.

    Returning None matters as much as succeeding: the caller falls back to
    generating each line separately rather than shipping mis-cut audio.
    """
    if len(texts) <= 1:
        return [wav] if texts else None

    needed = len(texts) - 1
    silences = find_silences(wav, sample_rate)
    if len(silences) < needed:
        return None

    cuts = sorted(int(centre) for centre, _ in silences[:needed])
    bounds = [0] + cuts + [wav.size]
    segments = [wav[bounds[i]:bounds[i + 1]] for i in range(len(texts))]

    if any(seg.size < sample_rate * 0.05 for seg in segments):
        return None

    # Each line should occupy roughly the share of time its length predicts.
    lengths = np.array([max(1, len(t)) for t in texts], dtype=float)
    expected = lengths / lengths.sum()
    actual = np.array([seg.size for seg in segments], dtype=float)
    actual /= actual.sum()
    if float(np.abs(expected - actual).max()) > MAX_SHARE_ERROR:
        return None

    return [trim_silence(seg, sample_rate) for seg in segments]


def trim_silence(
    wav: np.ndarray, sample_rate: int, floor_db: float = FLOOR_DB
) -> np.ndarray:
    """Drop leading/trailing quiet so assembled pauses are the ones we choose."""
    window = int(sample_rate * WINDOW_S)
    if window <= 0 or wav.size < window:
        return wav

    count = wav.size // window
    frames = wav[: count * window].reshape(count, window)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-10
    peak = float(np.abs(wav).max()) + 1e-10
    loud = np.where(20 * np.log10(rms / peak) >= floor_db)[0]
    if not len(loud):
        return wav

    start = max(0, (loud[0] - 1) * window)
    end = min(wav.size, (loud[-1] + 2) * window)
    return wav[start:end]
