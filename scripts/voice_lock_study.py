"""Does voice-locking actually hold a character steady across an interleaved scene?

Compares the two dialogue strategies on the same script:
  per-line  — every line its own generation (what the app did before)
  locked    — each speaker's lines rendered in one pass, then split apart

Metric is drift: the mean pairwise timbre distance between a character's own
lines. Lower is steadier. Reported against the distance between the two
characters, which is the scale that matters — drift approaching that gap is what
"the voices keep changing" sounds like.

Run:  .venv/bin/python scripts/voice_lock_study.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.segment import split_on_silence  # noqa: E402
from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

SCENE = [
    ("NARRATOR", "The door opened before he finished knocking."),
    ("VILLAIN", "You're late, and I don't like waiting."),
    ("NARRATOR", "Rain had been falling since noon."),
    ("VILLAIN", "Put the case down and step away from it."),
    ("NARRATOR", "Nobody in that building had seen him for a week."),
    ("VILLAIN", "Ask me that again and see what happens."),
]
VOICES = {"NARRATOR": "nature-documentary", "VILLAIN": "cheerful-kids-host"}
REPEATS = 3


def _frames(x, n=1024, hop=256):
    if x.size < n:
        return np.zeros((0, n), dtype=np.float32)
    c = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(c)[:, None]
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


def distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / denom) if denom > 1e-9 else 1.0


def drift(specs):
    pairs = [distance(specs[i], specs[j])
             for i in range(len(specs)) for j in range(i + 1, len(specs))]
    return float(np.mean(pairs)) if pairs else 0.0


def main() -> int:
    templates = load_templates()
    instructs = {s: templates.get(t).instruct for s, t in VOICES.items()}
    loaded = load_model()
    engine = SynthEngine(loaded)
    sr = loaded.sample_rate

    def gen(text, instruct, seed):
        return engine.synthesize(SynthRequest(
            text=text, instruct=instruct, language="English", seed=seed)).waveform

    per_line_drift, locked_drift, gaps, split_ok = [], [], [], 0

    for run in range(REPEATS):
        seed = 500 + run

        # Strategy A: one generation per line.
        by_speaker_a: dict[str, list] = {}
        for speaker, text in SCENE:
            by_speaker_a.setdefault(speaker, []).append(
                ltas(gen(text, instructs[speaker], seed))
            )

        # Strategy B: one generation per speaker, split back into lines.
        by_speaker_b: dict[str, list] = {}
        for speaker in VOICES:
            texts = [t for s, t in SCENE if s == speaker]
            merged = gen(" ".join(texts), instructs[speaker], seed)
            parts = split_on_silence(merged, sr, texts)
            if parts is None:
                parts = [gen(t, instructs[speaker], seed) for t in texts]
            else:
                split_ok += 1
            by_speaker_b[speaker] = [ltas(p) for p in parts]

        per_line_drift.append(np.mean([drift(v) for v in by_speaker_a.values()]))
        locked_drift.append(np.mean([drift(v) for v in by_speaker_b.values()]))
        names = list(VOICES)
        gaps.append(float(np.mean([
            distance(a, b)
            for a in by_speaker_b[names[0]] for b in by_speaker_b[names[1]]
        ])))
        print(f"  run {run + 1}: per-line {per_line_drift[-1]:.4f}   "
              f"locked {locked_drift[-1]:.4f}   gap {gaps[-1]:.4f}")

    per_line = float(np.mean(per_line_drift))
    locked = float(np.mean(locked_drift))
    gap = float(np.mean(gaps))

    print(f"\n{'strategy':12} {'drift':>8} {'drift/gap':>10}")
    print("-" * 32)
    print(f"{'per-line':12} {per_line:8.4f} {per_line / gap:10.2f}")
    print(f"{'locked':12} {locked:8.4f} {locked / gap:10.2f}")
    print(f"\ncharacter separation (gap): {gap:.4f}")
    print(f"splits succeeded: {split_ok}/{REPEATS * len(VOICES)}")
    improvement = (1 - locked / per_line) * 100 if per_line else 0
    print(f"voice-locking reduces drift by {improvement:.0f}%")
    print(
        "Locking holds a character measurably steadier."
        if locked < per_line * 0.95
        else "No measurable gain — drift is within the metric's noise floor."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
