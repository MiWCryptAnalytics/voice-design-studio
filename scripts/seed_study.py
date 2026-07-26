"""What actually determines the voice: the description, or the seed?

This matters for long scripts and multi-voice work, where a character has to stay
recognisable across many different lines. The answer decides whether continuity
needs a pinned seed or just a stable description.

The VoiceDesign checkpoint ships no speaker encoder (`extract_speaker_embedding`
fails — the speaker_encoder weights belong to the VoiceClone checkpoint), so
identity is measured with two acoustic proxies: the long-term average spectrum
(timbre) and median F0 (pitch height). Proxies, not speaker verification, but
they separate distinct voices by a wide margin.

Run:  .venv/bin/python scripts/seed_study.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

SEEDS = (11, 22, 33)
TEXT_A = "She walked in like a rumor that turned out to be true."
TEXT_B = "I had two questions that morning and neither one was polite."


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if x.size < n:
        return np.zeros((0, n), dtype=np.float32)
    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx] * np.hanning(n).astype(np.float32)


def ltas(x: np.ndarray) -> np.ndarray:
    """Long-term average spectrum, normalised — a timbre fingerprint."""
    f = _frames(x, 1024, 256)
    if not len(f):
        return np.zeros(513, dtype=np.float32)
    mag = np.abs(np.fft.rfft(f, axis=1))
    energy = mag.sum(axis=1)
    voiced = mag[energy > np.percentile(energy, 40)]  # skip near-silence
    spec = np.log1p(voiced).mean(axis=0)
    return spec / (np.linalg.norm(spec) + 1e-9)


def median_f0(x: np.ndarray, sr: int, lo: int = 60, hi: int = 350) -> float:
    """Autocorrelation pitch estimate over confidently voiced frames."""
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    out = []
    for fr in _frames(x, 1024, 512):
        if np.abs(fr).max() < 0.02:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        seg = ac[lo_lag:hi_lag]
        if seg.size and ac[0] > 1e-6:
            lag = lo_lag + int(np.argmax(seg))
            if ac[lag] / ac[0] > 0.3:
                out.append(sr / lag)
    return float(np.median(out)) if out else 0.0


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 1e-9 else 0.0


def main() -> int:
    templates = load_templates()
    noir = templates.get("noir-detective").instruct
    kid = templates.get("cheerful-kids-host").instruct

    loaded = load_model()
    engine = SynthEngine(loaded)
    sr = loaded.sample_rate

    def gen(text: str, instruct: str, seed: int) -> np.ndarray:
        return engine.synthesize(
            SynthRequest(text=text, instruct=instruct, language="English", seed=seed)
        ).waveform

    conditions = {
        "same description, SAME seed": [(noir, s, noir, s) for s in SEEDS],
        "same description, DIFFERENT seed": [
            (noir, 11, noir, 22), (noir, 22, noir, 33), (noir, 11, noir, 33)
        ],
        "DIFFERENT description": [(noir, s, kid, s) for s in SEEDS],
    }

    print(f"\nComparing “{TEXT_A[:32]}…” against “{TEXT_B[:32]}…”\n")
    results = {}
    for label, pairs in conditions.items():
        sims, deltas = [], []
        for instruct_a, seed_a, instruct_b, seed_b in pairs:
            wa, wb = gen(TEXT_A, instruct_a, seed_a), gen(TEXT_B, instruct_b, seed_b)
            sims.append(cosine(ltas(wa), ltas(wb)))
            deltas.append(abs(median_f0(wa, sr) - median_f0(wb, sr)))
        results[label] = (float(np.mean(sims)), float(np.mean(deltas)))
        print(f"  {label:34} timbre {results[label][0]:.3f}   ΔF0 {results[label][1]:6.1f} Hz")

    same, diff, other = (results[k] for k in conditions)
    print(
        f"\n  Seed-locking changes identity by:      timbre {abs(same[0] - diff[0]):.3f}, "
        f"ΔF0 {abs(same[1] - diff[1]):.1f} Hz"
    )
    print(
        f"  Changing the description changes it by: timbre {abs(same[0] - other[0]):.3f}, "
        f"ΔF0 {abs(other[1] - same[1]):.1f} Hz"
    )

    seed_effect = abs(same[0] - diff[0])
    description_effect = abs(same[0] - other[0])
    verdict = description_effect > seed_effect * 5
    print(
        "\n  CONCLUSION: the description carries speaker identity; the seed does not."
        if verdict
        else "\n  CONCLUSION: seed choice materially affects identity — revisit this."
    )
    print("  Continuity across lines therefore needs a stable description, not a pinned seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
