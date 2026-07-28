"""Why chunk joins are audible in a long single-voice script, and what fixes them.

Findings this encodes:
  * Timbre barely drifts between chunks (~0.02, below the metric's ~0.07 floor)
    and level varies by well under a decibel. Neither is the seam.
  * Each chunk is an independent utterance, so it ends on a terminal pitch fall
    and the next starts on a pitch reset. That doubles the pitch discontinuity
    at a boundary compared with the same boundary inside one pass.
  * Therefore the fix is fewer chunks, not better joining.

Run:  .venv/bin/python scripts/seam_study.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.script import DEFAULT_MAX_CHARS, split_script  # noqa: E402
from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

SCRIPT = (
    "The archive sits three floors below the street, behind a door that has "
    "no handle on the outside. Inside, the air is kept dry enough that paper "
    "from the last century still turns without cracking. Every shelf is "
    "numbered, and every number appears in a ledger that nobody has digitised. "
    "The woman who runs it has worked there for thirty-one years. She can find "
    "any document you name in under four minutes, and she has never once "
    "explained how."
)


def f0_track(x, sr, lo=60, hi=350, hop=0.02):
    n, step = int(sr * 0.04), int(sr * hop)
    lo_lag, hi_lag = int(sr / hi), int(sr / lo)
    out = []
    for s in range(0, max(0, x.size - n), step):
        fr = x[s:s + n].astype(np.float64)
        if np.abs(fr).max() < 0.02:
            out.append(0.0)
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[len(fr) - 1:]
        seg = ac[lo_lag:hi_lag]
        if seg.size and ac[0] > 1e-9:
            lag = lo_lag + int(np.argmax(seg))
            out.append(sr / lag if ac[lag] / ac[0] > 0.3 else 0.0)
        else:
            out.append(0.0)
    return np.array(out)


def voiced_mean(track):
    v = track[track > 0]
    return float(v.mean()) if v.size else 0.0


def ltas(x):
    n, hop = 1024, 256
    if x.size < n:
        return np.zeros(513, dtype=np.float32)
    c = 1 + (x.size - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(c)[:, None]
    f = x[idx] * np.hanning(n).astype(np.float32)
    mag = np.abs(np.fft.rfft(f, axis=1))
    e = mag.sum(axis=1)
    v = mag[e > np.percentile(e, 40)]
    s = np.log1p(v).mean(axis=0)
    return s / (np.linalg.norm(s) + 1e-9)


def main() -> int:
    templates = load_templates()
    instruct = templates.get("audiobook-storyteller").instruct
    loaded = load_model()
    engine = SynthEngine(loaded)
    sr = loaded.sample_rate

    def gen(text):
        return engine.synthesize(SynthRequest(
            text=text, instruct=instruct, language="English", seed=99)).waveform

    print(f"script: {len(SCRIPT)} characters")
    print(f"chunk limit is now {DEFAULT_MAX_CHARS} -> "
          f"{len(split_script(SCRIPT))} chunk(s); at the old 300 it was "
          f"{len(split_script(SCRIPT, 300))}\n")

    old_chunks = split_script(SCRIPT, 300)
    waves = [gen(c) for c in old_chunks]

    print("=== pitch discontinuity ===")
    jumps = []
    for i in range(len(waves) - 1):
        a = voiced_mean(f0_track(waves[i][-int(0.6 * sr):], sr))
        b = voiced_mean(f0_track(waves[i + 1][: int(0.6 * sr)], sr))
        jumps.append(abs(b - a))
        print(f"  across join {i + 1}: {a:6.1f} -> {b:6.1f} Hz   jump {abs(b - a):5.1f} Hz")

    whole = gen(SCRIPT)
    track = f0_track(whole, sr)
    win = int(sr * 0.02)
    n = whole.size // win
    rms = np.sqrt((whole[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-10
    db = 20 * np.log10(rms / (np.abs(whole).max() + 1e-10))
    spans, start = [], None
    for i, q in enumerate(db < -38):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 0.02 > 0.11 and 0.25 < start * 0.02 < whole.size / sr - 0.25:
                spans.append((start, i))
            start = None
    spans.sort(key=lambda p: -(p[1] - p[0]))
    inner = []
    for a, b in sorted(spans[: max(1, len(waves) - 1)]):
        before = voiced_mean(track[max(0, a - 30):a])
        after = voiced_mean(track[b:b + 30])
        inner.append(abs(after - before))
        print(f"  same boundary inside one pass: {before:6.1f} -> {after:6.1f} Hz   "
              f"jump {abs(after - before):5.1f} Hz")

    print("\n=== what is NOT the cause ===")
    specs = [ltas(w) for w in waves]
    pairs = [1 - float(np.dot(specs[i], specs[j]))
             for i in range(len(specs)) for j in range(i + 1, len(specs))]
    rmss = [float(np.sqrt((w ** 2).mean())) for w in waves]
    print(f"  timbre drift between chunks : {np.mean(pairs):.4f} "
          f"(metric noise floor ≈ 0.07)")
    print(f"  level spread between chunks : "
          f"{20 * np.log10(max(rmss) / min(rmss)):.2f} dB")

    if jumps and inner:
        print(f"\njoin pitch jump {np.mean(jumps):.1f} Hz vs "
              f"{np.mean(inner):.1f} Hz within one pass "
              f"({np.mean(jumps) / max(np.mean(inner), 1e-6):.1f}x)")
    print(
        "The seam is prosodic, not tonal: every chunk restarts the phrase contour.\n"
        "Fewer, longer passes is the only real remedy — hence the raised chunk limit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
