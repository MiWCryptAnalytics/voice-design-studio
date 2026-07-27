"""Regression test: a cached model must load and generate with no internet.

Two things in the stack reach for the Hub even when everything is cached — the
tokenizer's Mistral-regex patch calls `model_info()` regardless of
`local_files_only`, and the model's `from_pretrained` fetches `speech_tokenizer/*`
unless given a directory. Both are avoided by loading from the local snapshot
path. This test blackholes the network to prove it stays that way.

Run:  .venv/bin/python scripts/offline_check.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.engine.loader import (  # noqa: E402
    MODEL_ID,
    OFFLINE_ENV,
    forced_offline,
    is_model_cached,
    missing_from_cache,
    resolve_local_path,
)

failures: list[str] = []

# Point every HTTP path at a closed port so any network attempt fails fast.
BLACKHOLE = {
    "HTTP_PROXY": "http://127.0.0.1:9",
    "HTTPS_PROXY": "http://127.0.0.1:9",
    "ALL_PROXY": "http://127.0.0.1:9",
    "HF_HUB_ETAG_TIMEOUT": "2",
    "HF_HUB_DOWNLOAD_TIMEOUT": "2",
    "no_proxy": "",
    "NO_PROXY": "",
}

CHILD = """
import sys, time
sys.path.insert(0, %r)
from voicestudio.engine import SynthEngine, SynthRequest, load_model
start = time.perf_counter()
loaded = load_model()
elapsed = time.perf_counter() - start
engine = SynthEngine(loaded)
result = engine.synthesize(SynthRequest(
    text="Working without a network connection.",
    instruct="A calm narrator with a warm voice.",
    language="English", seed=1,
))
print("OFFLINE_OK", loaded.offline, round(elapsed, 2), round(result.duration, 2))
""" % str(ROOT)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def run_blackholed(extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **BLACKHOLE, **(extra_env or {})}
    return subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-c", CHILD],
        capture_output=True, text=True, env=env, timeout=900,
    )


def main() -> int:
    print("--- cache inspection (no network) ---")
    missing = missing_from_cache(MODEL_ID)
    check("model is fully cached", not missing, f"missing: {missing}" if missing else "")
    if missing:
        print("\nCannot test offline loading without a complete cache.")
        return 1

    path = resolve_local_path(MODEL_ID)
    check("local snapshot path resolves", path is not None and Path(path).is_dir(), str(path))
    check("snapshot really holds the weights",
          path is not None and (Path(path) / "model.safetensors").exists()
          and (Path(path) / "speech_tokenizer" / "model.safetensors").exists())
    check("an uncached repo reports everything missing",
          len(missing_from_cache("definitely/not-a-real-model")) == 9)
    check("a local directory resolves to itself",
          resolve_local_path(str(ROOT)) == str(ROOT))

    print("\n--- loading with the network blackholed ---")
    proc = run_blackholed()
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("OFFLINE_OK")), "")
    check("loads and generates with no internet", bool(line),
          (proc.stderr.strip().splitlines() or ["no output"])[-1] if not line else "")
    if line:
        _, offline, elapsed, duration = line.split()
        check("reported as an offline load", offline == "True")
        check("load was fast (no network waits)", float(elapsed) < 60,
              f"{elapsed}s")
        check("audio was actually produced", float(duration) > 0.5, f"{duration}s")

    print(f"\n--- {OFFLINE_ENV} ---")
    check("forced_offline is off by default", not forced_offline())
    proc = run_blackholed({OFFLINE_ENV: "1"})
    check(f"{OFFLINE_ENV}=1 still loads from cache",
          any(ln.startswith("OFFLINE_OK") for ln in proc.stdout.splitlines()),
          (proc.stderr.strip().splitlines() or ["no output"])[-1])

    check("is_model_cached agrees", is_model_cached(MODEL_ID))

    print("\nOFFLINE CHECK", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
