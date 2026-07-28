"""Audio helpers: WAV IO, waveform envelopes, concatenation, normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def write_wav(path: Path | str, wav: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate)
    return path


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def peak_normalize(wav: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak <= 1e-9:
        return wav
    return (wav * (target / peak)).astype(np.float32)


def concat(
    chunks: list[np.ndarray], sample_rate: int, gap_seconds: float = 0.25
) -> np.ndarray:
    """Join chunks with silence between them."""
    chunks = [c for c in chunks if c is not None and c.size]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(max(0.0, gap_seconds) * sample_rate), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(c.astype(np.float32))
    return np.concatenate(out)


def concat_crossfade(
    pieces: list[np.ndarray],
    sample_rate: int,
    gap_seconds: float = 0.0,
    fade_ms: float = 15.0,
) -> np.ndarray:
    """Join pieces with a short crossfade so the seam isn't a hard splice.

    Generated chunks already end and begin with their own near-silence, which
    carries faint room tone. Butting them against inserted digital zero leaves an
    audible hole, so overlap the boundary slightly instead.
    """
    pieces = [p for p in pieces if p is not None and p.size]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    if len(pieces) == 1:
        return pieces[0].astype(np.float32)

    fade = max(0, int(sample_rate * fade_ms / 1000.0))
    gap = np.zeros(int(max(0.0, gap_seconds) * sample_rate), dtype=np.float32)

    out = pieces[0].astype(np.float32)
    for piece in pieces[1:]:
        piece = piece.astype(np.float32)
        if gap.size:
            out = np.concatenate([out, gap])
        overlap = min(fade, out.size, piece.size)
        if overlap <= 0:
            out = np.concatenate([out, piece])
            continue
        # Equal-power crossfade keeps perceived level steady through the seam.
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        blended = out[-overlap:] * np.cos(ramp * np.pi / 2) + piece[:overlap] * np.sin(
            ramp * np.pi / 2
        )
        out = np.concatenate([out[:-overlap], blended, piece[overlap:]])
    return out


def envelope(wav: np.ndarray, buckets: int) -> np.ndarray:
    """Downsample to per-bucket peak magnitudes for waveform drawing.

    Returns `buckets` values in 0..1. Peak (not mean) keeps transients visible
    at any zoom level.
    """
    buckets = max(1, int(buckets))
    if wav is None or wav.size == 0:
        return np.zeros(buckets, dtype=np.float32)

    mag = np.abs(wav).astype(np.float32)
    if mag.size < buckets:
        out = np.interp(
            np.linspace(0, mag.size - 1, buckets),
            np.arange(mag.size),
            mag,
        ).astype(np.float32)
    else:
        usable = (mag.size // buckets) * buckets
        out = mag[:usable].reshape(buckets, -1).max(axis=1)
        if usable < mag.size:
            out[-1] = max(out[-1], float(mag[usable:].max()))

    top = float(out.max())
    return out / top if top > 1e-9 else out


def duration_of(wav: np.ndarray, sample_rate: int) -> float:
    return len(wav) / float(sample_rate or 1)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
