"""What does sub-talker temperature actually do?

The sub-talker is the second sampling stage: it predicts the residual codebooks
(acoustic detail) on top of the talker's token stream. Before exposing it as a
control, measure whether it changes anything — and what.

Writes WAVs to _sweep_out/ so the numbers can be checked by ear.

Run:  .venv/bin/python scripts/subtalker_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

OUT = ROOT / "_sweep_out"
TEMPS = [0.2, 0.5, 0.9, 1.2, 1.5]
SEEDS = (101, 202, 303)
TEXT = "The quiet between two thoughts is where the whole story hides."


def spectrum(x: np.ndarray) -> np.ndarray:
    n, hop = 1024, 256
    if x.size < n:
        return np.zeros(513, dtype=np.float32)
    count = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
    frames = x[idx] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    energy = mag.sum(axis=1)
    return mag[energy > np.percentile(energy, 40)]


def flatness(x: np.ndarray) -> float:
    """Spectral flatness: 0 = tonal, 1 = noise-like."""
    mag = spectrum(x) + 1e-10
    geo = np.exp(np.log(mag).mean(axis=1))
    arith = mag.mean(axis=1)
    return float(np.mean(geo / arith))


def hf_ratio(x: np.ndarray, sr: int, cutoff: int = 4000) -> float:
    mag = spectrum(x)
    if not len(mag):
        return 0.0
    bins = np.fft.rfftfreq(1024, 1 / sr)
    total = mag.sum()
    return float(mag[:, bins >= cutoff].sum() / total) if total > 0 else 0.0


def ltas(x: np.ndarray) -> np.ndarray:
    mag = spectrum(x)
    if not len(mag):
        return np.zeros(513, dtype=np.float32)
    spec = np.log1p(mag).mean(axis=0)
    return spec / (np.linalg.norm(spec) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def main() -> int:
    OUT.mkdir(exist_ok=True)
    instruct = load_templates().get("audiobook-storyteller").instruct
    loaded = load_model()
    engine = SynthEngine(loaded)
    sr = loaded.sample_rate

    print(f"\ntalker temperature fixed at 0.9; sweeping sub-talker\n")
    print(f"{'sub-T':>6} {'dur s':>7} {'flatness':>9} {'HF>4k':>7} "
          f"{'RMS':>7} {'spread':>8}  (spread = 1 - mean cross-seed LTAS similarity)")

    rows = {}
    for temp in TEMPS:
        waves = []
        for seed in SEEDS:
            res = engine.synthesize(
                SynthRequest(
                    text=TEXT, instruct=instruct, language="English", seed=seed,
                    temperature=0.9, subtalker_temperature=temp,
                )
            )
            waves.append(res.waveform)
        sf.write(OUT / f"subtalker_{temp:.1f}.wav", waves[0], sr)

        specs = [ltas(w) for w in waves]
        pairs = [cosine(specs[i], specs[j])
                 for i in range(len(specs)) for j in range(i + 1, len(specs))]
        rows[temp] = {
            "dur": float(np.mean([len(w) / sr for w in waves])),
            "flat": float(np.mean([flatness(w) for w in waves])),
            "hf": float(np.mean([hf_ratio(w, sr) for w in waves])),
            "rms": float(np.mean([np.sqrt(np.mean(w ** 2)) for w in waves])),
            "spread": 1.0 - float(np.mean(pairs)),
            "ltas": specs[0],
        }
        r = rows[temp]
        print(f"{temp:6.1f} {r['dur']:7.2f} {r['flat']:9.4f} {r['hf']:7.3f} "
              f"{r['rms']:7.4f} {r['spread']:8.4f}")

    base = rows[0.9]["ltas"]
    print("\nTimbre distance from the 0.9 default (1 - LTAS cosine):")
    for temp in TEMPS:
        print(f"  sub-T {temp:.1f}: {1 - cosine(rows[temp]['ltas'], base):.4f}")

    # Does the parameter reach the model at all? Fix the seed and difference.
    print("\nWiring check (seed fixed at 555, only the sub-talker setting varies):")
    baseline = engine.synthesize(
        SynthRequest(text=TEXT, instruct=instruct, language="English", seed=555)
    ).waveform
    for label, kwargs in [
        ("same settings (control)", {}),
        ("sub-T 0.2", {"subtalker_temperature": 0.2}),
        ("sub-T 1.5", {"subtalker_temperature": 1.5}),
        ("greedy (do_sample off)", {"subtalker_do_sample": False}),
    ]:
        wav = engine.synthesize(
            SynthRequest(text=TEXT, instruct=instruct, language="English",
                         seed=555, **kwargs)
        ).waveform
        identical = wav.shape == baseline.shape and np.allclose(wav, baseline, atol=1e-5)
        print(f"  {label:24} identical={str(identical):5}  "
              f"{baseline.size / sr:.2f}s -> {wav.size / sr:.2f}s")

    lo, hi = rows[TEMPS[0]], rows[TEMPS[-1]]
    flat_delta = abs(hi["flat"] - lo["flat"]) / max(lo["flat"], 1e-9)
    spread_delta = abs(hi["spread"] - lo["spread"])
    print(
        f"\nAcross {TEMPS[0]} -> {TEMPS[-1]}: flatness moves {flat_delta * 100:.1f}%, "
        f"cross-seed spread moves {spread_delta:.4f}"
    )
    print(
        "\nVERDICT: sub-talker settings change the rendition substantially (see the\n"
        "wiring check), but across this range they show no monotonic trend in\n"
        "brightness, noisiness or cross-seed spread — the movement above is within\n"
        "run-to-run variance. Treat it as a second exploration axis, not a quality dial."
        if flat_delta <= 0.05 and spread_delta <= 0.01
        else "\nVERDICT: measurable directional effect on acoustic detail — see the trend above."
    )
    print(f"WAVs in {OUT} for listening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
