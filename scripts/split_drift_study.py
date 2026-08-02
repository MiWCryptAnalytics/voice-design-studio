"""Why does "Split long scripts" change the voice between chunks?

Renders the same narration under one description and one seed, three ways:

  single-pass    : the whole text in one generation, sliced into pieces after
  split          : one generation per chunk — what "Split long scripts" does
  split+profile  : one generation per chunk, conditioned on a fixed speaker
                   embedding (a voice profile) instead of the description

Speaker identity is measured with the Base checkpoint's own speaker encoder:
cosine similarity between x-vectors of the pieces (1.0 = same speaker). This is
an identity metric, unlike the LTAS timbre fingerprint used by
consistency_study.py, which tracks spectral shape and can stay small while the
perceived speaker changes.

Run:  .venv/bin/python scripts/split_drift_study.py
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core.script import split_script  # noqa: E402
from voicestudio.engine import (  # noqa: E402
    BASE_MODEL_ID,
    SynthEngine,
    SynthRequest,
    load_model,
)

INSTRUCT = (
    "A calm, deep-voiced male documentary narrator in his fifties, speaking "
    "slowly and evenly with a warm, reassuring tone."
)

# ~800 characters of sentence-rich narration; split at 160 chars to force the
# same per-chunk generation the app performs at 1500 on a longer script.
TEXT = (
    "The river begins as a trickle of meltwater high above the treeline. "
    "By the time it reaches the valley floor it has gathered the strength of a "
    "hundred hidden springs. "
    "Along its banks, willows lean into the current as if listening. "
    "Each spring the floodwater redraws the map, carving new channels through "
    "the gravel beds. "
    "The herons return in the last week of March, always to the same three "
    "nests. "
    "Farther downstream, the water slows and darkens, heavy with the silt of "
    "the mountains it has left behind. "
    "At dusk the surface turns to hammered copper, and the first bats skim "
    "insects from the air. "
    "It is an old landscape, patient and unhurried, and the river is the "
    "oldest thing in it."
)

SEED = 20260728
CHUNK_CHARS = 160


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(label: str, embs: list[np.ndarray]) -> None:
    pairs = [cos(a, b) for i, a in enumerate(embs) for b in embs[i + 1 :]]
    adjacent = [cos(a, b) for a, b in zip(embs, embs[1:])]
    print(
        f"{label:24s} mean {np.mean(pairs):.3f}   min {np.min(pairs):.3f}   "
        f"worst adjacent seam {np.min(adjacent):.3f}"
    )


def main() -> int:
    design = SynthEngine(load_model(progress=print))
    base = SynthEngine(load_model(model_id=BASE_MODEL_ID, progress=print))

    chunks = split_script(TEXT, CHUNK_CHARS)
    print(f"\n{len(TEXT)} chars -> {len(chunks)} chunks at max {CHUNK_CHARS}\n")

    req = SynthRequest(text="", instruct=INSTRUCT, language="English", seed=SEED)

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        return base.extract_profile(wav, sr, name="probe").embedding

    # Split off: one continuous generation, then measure its internal
    # consistency by slicing it into as many pieces as there are chunks.
    single = design.synthesize(replace(req, text=TEXT))
    print(f"single pass: {single.duration:.1f}s, truncated={single.truncated}")
    slices = np.array_split(single.waveform, len(chunks))
    single_embs = [embed(s, single.sample_rate) for s in slices]

    # Split on: per-chunk generation, same description and seed every time —
    # exactly what _run_joined does per item.
    split_takes = [design.synthesize(replace(req, text=c)) for c in chunks]
    for i, t in enumerate(split_takes):
        print(f"chunk {i + 1}: {t.duration:.1f}s")
    split_embs = [embed(t.waveform, t.sample_rate) for t in split_takes]

    # Split on, but conditioned on a fixed embedding taken from the
    # single-pass take — the app's voice-profile path.
    profile = base.extract_profile(single.waveform, single.sample_rate, name="study")
    prof_req = replace(
        req, voice_profile=profile, temperature=0.7, subtalker_temperature=0.7
    )
    prof_takes = [base.synthesize(replace(prof_req, text=c)) for c in chunks]
    prof_embs = [embed(t.waveform, t.sample_rate) for t in prof_takes]

    print("\nspeaker similarity (x-vector cosine, 1.0 = same speaker):")
    report("single-pass slices", single_embs)
    report("split chunks", split_embs)
    report("split + voice profile", prof_embs)

    print("\nsimilarity of each piece to the first piece:")
    for label, embs in [
        ("single-pass", single_embs),
        ("split", split_embs),
        ("split+profile", prof_embs),
    ]:
        sims = " ".join(f"{cos(embs[0], e):.3f}" for e in embs[1:])
        print(f"{label:24s} {sims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
