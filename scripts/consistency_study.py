"""How stable is a character's voice across many lines, and what stabilises it?

The dialogue feature assumes a stable description keeps a character recognisable.
That holds far better than changing the description — but "better" isn't "stable",
and drift across a long scene is audible.

Measures, per condition:
  spread  = mean pairwise timbre distance between a character's own lines (low = stable)
  gap     = timbre distance between the two characters (high = distinct)
  ratio   = spread / gap. Below ~0.25 the characters stay clearly apart; as it
            approaches 1 a character's own lines drift as much as the two
            characters differ, which is what "the voices keep changing" sounds like.

Run:  .venv/bin/python scripts/consistency_study.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

LINES_A = [
    "The door opened before I finished knocking.",
    "Rain had been falling since noon and showed no sign of stopping.",
    "She left the envelope on the table without a word.",
    "Nobody in that building had seen him for a week.",
    "I counted the exits out of habit, not fear.",
]
LINES_B = [
    "You're late, and I don't like waiting.",
    "Put the case down and step away from it.",
    "There were three of us at the start of this.",
    "Ask me that again and see what happens.",
    "I told you the price before you agreed.",
]

CONDITIONS = [
    ("baseline  T0.9 p1.0", dict(temperature=0.9, top_p=1.0)),
    ("T0.7", dict(temperature=0.7, top_p=1.0)),
    ("T0.5", dict(temperature=0.5, top_p=1.0)),
    ("T0.3", dict(temperature=0.3, top_p=1.0)),
    ("T0.7 p0.8", dict(temperature=0.7, top_p=0.8)),
    ("T0.5 p0.8 subT0.5", dict(temperature=0.5, top_p=0.8,
                               subtalker_temperature=0.5, subtalker_top_p=0.8)),
]


def _frames(x, n=1024, hop=256):
    if x.size < n:
        return np.zeros((0, n), dtype=np.float32)
    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    return x[idx] * np.hanning(n).astype(np.float32)


def ltas(x):
    f = _frames(x)
    if not len(f):
        return np.zeros(513, dtype=np.float32)
    mag = np.abs(np.fft.rfft(f, axis=1))
    energy = mag.sum(axis=1)
    voiced = mag[energy > np.percentile(energy, 40)]
    spec = np.log1p(voiced).mean(axis=0)
    return spec / (np.linalg.norm(spec) + 1e-9)


def median_f0(x, sr, lo=60, hi=350):
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


def distance(a, b):
    """1 - cosine similarity of the timbre fingerprints."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-9 else 1.0


def spread_of(specs):
    pairs = [distance(specs[i], specs[j])
             for i in range(len(specs)) for j in range(i + 1, len(specs))]
    return float(np.mean(pairs)) if pairs else 0.0


def main() -> int:
    templates = load_templates()
    voice_a = templates.get("nature-documentary").instruct
    voice_b = templates.get("sinister-villain").instruct

    loaded = load_model()
    engine = SynthEngine(loaded)
    sr = loaded.sample_rate

    print(f"\n{'condition':22} {'spread':>8} {'gap':>8} {'ratio':>7} "
          f"{'F0 sd A':>8} {'F0 sd B':>8}")
    print("-" * 66)

    results = {}
    for label, params in CONDITIONS:
        specs_a, specs_b, f0_a, f0_b = [], [], [], []
        for lines, instruct, specs, f0s in (
            (LINES_A, voice_a, specs_a, f0_a),
            (LINES_B, voice_b, specs_b, f0_b),
        ):
            for i, line in enumerate(lines):
                wav = engine.synthesize(SynthRequest(
                    text=line, instruct=instruct, language="English",
                    seed=1000 + i, **params,
                )).waveform
                specs.append(ltas(wav))
                f0s.append(median_f0(wav, sr))

        spread = (spread_of(specs_a) + spread_of(specs_b)) / 2
        gap = float(np.mean([distance(a, b) for a in specs_a for b in specs_b]))
        ratio = spread / gap if gap > 1e-9 else float("inf")
        results[label] = (spread, gap, ratio)
        print(f"{label:22} {spread:8.4f} {gap:8.4f} {ratio:7.2f} "
              f"{np.std(f0_a):8.1f} {np.std(f0_b):8.1f}")

    base = results[CONDITIONS[0][0]]
    best = min(results.items(), key=lambda kv: kv[1][2])
    print(
        f"\nbaseline ratio {base[2]:.2f} -> best '{best[0]}' ratio {best[1][2]:.2f}"
        f"  ({(1 - best[1][2] / base[2]) * 100:.0f}% tighter)"
    )
    print(
        "Lower sampling temperature is the lever: it narrows how far each line can\n"
        "wander from the description while leaving the characters just as distinct."
        if best[1][2] < base[2] * 0.9
        else "No setting meaningfully tightened consistency — the drift is elsewhere."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
