"""Headless engine check: load the model, synthesize two voices, write WAVs.

Run:  .venv/bin/python scripts/smoke_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "_smoke_out"
SENTENCE = "The quiet between two thoughts is where the whole story hides."

CASES = [
    ("documentary", "An older British man with a warm, gravelly baritone, speaking "
                    "slowly and with hushed reverence, as if narrating footage of "
                    "animals at dawn."),
    ("commercial", "A bright, energetic young woman with a clear, punchy voice, "
                   "speaking quickly with a confident smile and rising enthusiasm."),
]


def main() -> int:
    OUT.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    loaded = load_model(progress=lambda m: print(f"  [load] {m}"))
    print(f"Loaded in {time.perf_counter() - t0:.1f}s | {loaded.device_label} | sr={loaded.sample_rate}")
    print(f"Languages: {', '.join(loaded.languages)}")

    engine = SynthEngine(loaded)
    ok = True

    for name, instruct in CASES:
        req = SynthRequest(text=SENTENCE, instruct=instruct, language="English", seed=1234)
        res = engine.synthesize(req)
        path = OUT / f"{name}.wav"
        sf.write(path, res.waveform, res.sample_rate)

        peak = float(np.abs(res.waveform).max()) if res.waveform.size else 0.0
        silent = peak < 1e-3
        too_short = res.duration < 0.5
        status = "FAIL" if (silent or too_short) else "ok"
        if status == "FAIL":
            ok = False
        print(
            f"[{status}] {name}: {res.duration:.2f}s  peak={peak:.3f}  "
            f"gen={res.elapsed:.2f}s  rtf={res.realtime_factor:.2f}x  -> {path.name}"
        )

    # Reproducibility: same seed must give identical audio.
    req = SynthRequest(text=SENTENCE, instruct=CASES[0][1], language="English", seed=777)
    a = engine.synthesize(req).waveform
    b = engine.synthesize(req).waveform
    same = a.shape == b.shape and np.allclose(a, b, atol=1e-5)
    print(f"[{'ok' if same else 'FAIL'}] reproducible with fixed seed: {same}")
    ok = ok and same

    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
