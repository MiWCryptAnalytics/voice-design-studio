"""Remaining plan verification: templates, presets, persistence, cancel, VRAM.

Run:  .venv/bin/python scripts/final_checks.py
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core import audio as audio_utils  # noqa: E402
from voicestudio.core.library import VoiceLibrary, VoicePreset  # noqa: E402
from voicestudio.core.templates import load_templates  # noqa: E402
from voicestudio.engine import SynthEngine, SynthRequest, load_model  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """How similar two takes are, length-aligned. 1.0 means identical."""
    n = min(a.size, b.size)
    if n < 100:
        return 0.0
    x, y = a[:n], b[:n]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(abs(np.dot(x, y)) / denom) if denom > 1e-9 else 0.0


def main() -> int:
    templates = load_templates()

    # ---- preset forking: editing a template must not alter the built-in ----
    lib = VoiceLibrary()  # in-memory, not the user's file
    original = templates.get("noir-detective")
    forked = VoicePreset(
        name="My Noir",
        instruct=original.instruct + " Slightly faster.",
        source_template=original.id,
    )
    lib._presets.append(forked)
    reloaded_template = load_templates().get("noir-detective")
    check(
        "editing a fork leaves the built-in template untouched",
        reloaded_template.instruct == original.instruct,
    )
    check("fork records its source template", forked.source_template == "noir-detective")

    # ---- library persistence round-trip ----
    real_lib = VoiceLibrary.load()
    before = len(real_lib.all())
    probe = real_lib.add(VoicePreset(name="__verify_probe__", instruct="test voice"))
    check("preset persisted", VoiceLibrary.load().get(probe.id) is not None)
    real_lib.remove(probe.id)
    check(
        "preset removed cleanly",
        VoiceLibrary.load().get(probe.id) is None
        and len(VoiceLibrary.load().all()) == before,
    )

    # ---- duplicate names get disambiguated rather than colliding ----
    lib2 = VoiceLibrary()
    lib2._presets.append(VoicePreset(name="Dup", instruct="a"))
    second = VoicePreset(name="Dup", instruct="b")
    second.name = lib2._unique_name(second.name)
    check("duplicate preset names disambiguated", second.name == "Dup (2)", second.name)

    # ---- engine-backed checks ----
    print("\nloading model…")
    loaded = load_model()
    engine = SynthEngine(loaded)
    sentence = templates.demo_sentence
    seed = templates.demo_seed

    # Template distinctness: same text + same seed, only the description differs.
    probe_ids = ["nature-documentary", "upbeat-commercial", "synthetic-ai", "guided-meditation"]
    waves = {}
    for tid in probe_ids:
        t = templates.get(tid)
        res = engine.synthesize(
            SynthRequest(text=sentence, instruct=t.instruct, language="English", seed=seed)
        )
        waves[tid] = res.waveform
        print(f"    {t.name:22} {res.duration:5.2f}s  peak={np.abs(res.waveform).max():.3f}")

    worst = 0.0
    worst_pair = ()
    for i, a in enumerate(probe_ids):
        for b in probe_ids[i + 1 :]:
            c = correlation(waves[a], waves[b])
            if c > worst:
                worst, worst_pair = c, (a, b)
    check(
        "templates produce audibly distinct voices",
        worst < 0.5,
        f"highest similarity {worst:.2f} between {worst_pair[0]} and {worst_pair[1]}",
    )
    check(
        "all template takes are non-silent",
        all(float(np.abs(w).max()) > 1e-3 for w in waves.values()),
    )

    # ---- a pinned preset must reproduce the exact voice it saved ----
    design = templates.get("late-night-radio").instruct
    first = engine.synthesize(
        SynthRequest(text=sentence, instruct=design, language="English", seed=None)
    )
    saved = VoiceLibrary.load()
    pinned = saved.add(
        VoicePreset(name="__verify_pinned__", instruct=design, seed=first.seed)
    )
    # Round-trip through disk, exactly as a restart would.
    restored = VoiceLibrary.load().get(pinned.id)
    check("pinned seed survives save/load", restored.seed == first.seed,
          f"{restored.seed} vs {first.seed}")
    again = engine.synthesize(
        SynthRequest(text=sentence, instruct=restored.instruct,
                     language="English", seed=restored.seed)
    )
    check(
        "reloaded preset reproduces the same voice",
        again.waveform.shape == first.waveform.shape
        and np.allclose(again.waveform, first.waveform, atol=1e-5),
    )

    # An unpinned preset is deliberately a family of voices, not one voice.
    unpinned = saved.add(VoicePreset(name="__verify_unpinned__", instruct=design))
    check("unpinned preset stores no seed",
          VoiceLibrary.load().get(unpinned.id).seed is None)
    drifted = engine.synthesize(
        SynthRequest(text=sentence, instruct=design, language="English", seed=None)
    )
    check(
        "unpinned regeneration gives a different realisation",
        not (drifted.waveform.shape == first.waveform.shape
             and np.allclose(drifted.waveform, first.waveform, atol=1e-5)),
    )

    # Presets written before seeds existed must still load.
    legacy = VoicePreset(**{"name": "legacy", "instruct": "old preset"})
    check("legacy presets default to unpinned", legacy.seed is None
          and not legacy.seed_pinned)

    for pid in (pinned.id, unpinned.id):
        saved.remove(pid)
    check("verification presets cleaned up",
          all(VoiceLibrary.load().get(p) is None for p in (pinned.id, unpinned.id)))

    # ---- max_new_tokens: the ceiling and its truncation flag ----
    tiny = engine.synthesize(
        SynthRequest(text=sentence, instruct=design, language="English",
                     seed=8, max_new_tokens=32)
    )
    check("tiny token budget is flagged as truncated", tiny.truncated,
          f"{tiny.codes} codes, {tiny.duration:.2f}s")
    roomy = engine.synthesize(
        SynthRequest(text=sentence, instruct=design, language="English", seed=8)
    )
    check("default budget is not flagged", not roomy.truncated,
          f"{roomy.codes} codes of {roomy.request.max_new_tokens}")
    check("truncated audio really is shorter", tiny.duration < roomy.duration,
          f"{tiny.duration:.2f}s vs {roomy.duration:.2f}s")
    check("default matches the checkpoint's own generate_config",
          roomy.request.max_new_tokens == int(
              loaded.model.generate_config.get("max_new_tokens", 0)),
          str(roomy.request.max_new_tokens))

    # ---- sub-talker parameters reach the model ----
    fixed = dict(text=sentence, instruct=design, language="English", seed=606)
    a = engine.synthesize(SynthRequest(**fixed, subtalker_temperature=0.9))
    b = engine.synthesize(SynthRequest(**fixed, subtalker_temperature=0.9))
    c = engine.synthesize(SynthRequest(**fixed, subtalker_temperature=0.2))
    d = engine.synthesize(SynthRequest(**fixed, subtalker_do_sample=False))
    check("identical sub-talker settings are reproducible",
          a.waveform.shape == b.waveform.shape
          and np.allclose(a.waveform, b.waveform, atol=1e-5))
    check("sub-talker temperature changes the output",
          not (a.waveform.shape == c.waveform.shape
               and np.allclose(a.waveform, c.waveform, atol=1e-5)))
    check("greedy sub-talker changes the output",
          not (a.waveform.shape == d.waveform.shape
               and np.allclose(a.waveform, d.waveform, atol=1e-5)))

    # ---- take history round-trips the new fields ----
    from voicestudio.core.history import Take, TakeHistory  # noqa: E402

    hist = TakeHistory.load()
    wav_path = hist.next_wav_path()
    audio_utils.write_wav(wav_path, c.waveform, c.sample_rate)
    probe_take = hist.add(Take(
        instruct=design, text=sentence, language="English", seed=606,
        wav_path=str(wav_path), sample_rate=c.sample_rate, duration=c.duration,
        elapsed=c.elapsed, subtalker_temperature=0.2, subtalker_top_k=7,
        subtalker_do_sample=False, truncated=True, spoken_text="normalized text",
    ))
    reloaded = TakeHistory.load().get(probe_take.id)
    check("sub-talker settings survive history save/load",
          reloaded.subtalker_temperature == 0.2 and reloaded.subtalker_top_k == 7
          and reloaded.subtalker_do_sample is False)
    check("truncation flag and spoken text persist",
          reloaded.truncated and reloaded.spoken_text == "normalized text")
    hist.remove(probe_take.id)
    check("probe take cleaned up", TakeHistory.load().get(probe_take.id) is None)

    # ---- cancellation ----
    from voicestudio.ui.workers import SCRIPT, EngineHost, SynthJob

    host = EngineHost()
    host._engine = engine
    produced: list = []
    host.takeReady.connect(produced.append)
    done: list[str] = []
    host.jobDone.connect(done.append)

    chunks = [f"This is chunk number {i}." for i in range(6)]
    job = SynthJob(
        request=SynthRequest(text="", instruct=templates.get("wise-elder").instruct,
                             language="English", seed=42),
        chunks=chunks,
        seeds=[42],
        mode=SCRIPT,
    )
    threading.Timer(3.0, host.cancel).start()
    host.run_job(job)
    check("cancel stops a multi-chunk job", done and done[0] == "cancelled", str(done))
    check("cancelled job emits no partial take", len(produced) == 0, f"{len(produced)} takes")

    # ---- VRAM stability ----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_reserved() / (1024**3)
        for _ in range(10):
            engine.synthesize(
                SynthRequest(text=sentence, instruct=original.instruct,
                             language="English", seed=None)
            )
        torch.cuda.synchronize()
        after = torch.cuda.memory_reserved() / (1024**3)
        growth = after - baseline
        check(
            "VRAM stable across 10 generations",
            growth < 1.0,
            f"{baseline:.2f} -> {after:.2f} GiB (+{growth:.2f})",
        )

    # ---- audio helpers on real audio ----
    joined = audio_utils.concat(list(waves.values()), loaded.sample_rate, 0.25)
    expected = sum(w.size for w in waves.values()) + 3 * int(0.25 * loaded.sample_rate)
    check("concat length correct", joined.size == expected, f"{joined.size} vs {expected}")
    check("normalize hits target peak",
          abs(float(np.abs(audio_utils.peak_normalize(joined)).max()) - 0.95) < 1e-3)

    print("\nFINAL CHECKS", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
