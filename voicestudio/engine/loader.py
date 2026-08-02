"""Load Qwen3-TTS VoiceDesign through the transformers Auto* API.

The `qwen3_tts` architecture is not part of upstream transformers; the `qwen_tts`
package supplies the config/model/processor classes, which we register into the
transformers Auto mappings so the model loads through the standard
`AutoModel.from_pretrained` path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

# Importing qwen_tts prints a "flash-attn is not installed" banner. It comes
# from the 25 Hz tokenizer's Whisper encoder, which the 12 Hz checkpoints this
# app loads never execute — harmless, see DESIGN.md.
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
# The Base checkpoint is the only one in the family with a speaker encoder, so
# voice-profile extraction and embedding-conditioned generation both need it.
# It shares the architecture above and loads through the same path.
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Everything the model and processor need. Checked before choosing offline, so a
# partially-downloaded cache doesn't turn into a confusing load failure.
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)

# Set to 1 to force local-only loading and fail loudly instead of reaching out.
OFFLINE_ENV = "VOICESTUDIO_OFFLINE"

# Fallback list; the loaded model is authoritative via get_supported_languages().
FALLBACK_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

_registered = False


def _register_once() -> None:
    """Register qwen3_tts with transformers' Auto mappings (idempotent)."""
    global _registered
    if _registered:
        return
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    _registered = True


def pick_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    # bf16 needs Ampere or newer; the 3090 qualifies.
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class LoadedModel:
    model: Qwen3TTSForConditionalGeneration
    processor: Qwen3TTSProcessor
    device: str
    dtype: torch.dtype
    sample_rate: int
    languages: list[str] = field(default_factory=list)
    offline: bool = False

    @property
    def device_label(self) -> str:
        if self.device.startswith("cuda"):
            idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
            return f"{torch.cuda.get_device_name(idx)} ({str(self.dtype).replace('torch.', '')})"
        return f"CPU ({str(self.dtype).replace('torch.', '')})"


def load_model(
    model_id: str = MODEL_ID,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
    local_files_only: bool | None = None,
) -> LoadedModel:
    """Load model + processor. `progress` receives human-readable status strings.

    When the model is fully cached the load runs entirely offline. Without this,
    every launch contacts the Hub — the model's own `from_pretrained` fetches the
    speech tokenizer and the processor revalidates against the API — so the app
    needs a working connection even though nothing has to be downloaded.
    """

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    say("Registering qwen3_tts with transformers…")
    _register_once()

    device = device or pick_device()
    dtype = pick_dtype(device)

    forced = forced_offline()
    local_path: str | None = None
    if local_files_only is None:
        missing = missing_from_cache(model_id)
        if missing and forced:
            raise RuntimeError(
                f"{OFFLINE_ENV} is set but the cache is incomplete. Missing: "
                + ", ".join(missing)
            )
        if missing:
            say(f"Downloading {len(missing)} missing file(s) from the Hub…")
        local_files_only = not missing

    if local_files_only:
        local_path = resolve_local_path(model_id)
        if local_path is None:
            if forced:
                raise RuntimeError(
                    f"{OFFLINE_ENV} is set but no cached snapshot was found for "
                    f"{model_id}."
                )
            local_files_only = False

    def build(offline: bool):
        # A local directory is what keeps the load off the network entirely.
        source = local_path if (offline and local_path) else model_id
        say(
            f"Loading weights onto {device} "
            f"({str(dtype).replace('torch.', '')})"
            + (" from local cache…" if offline else "…")
        )
        loaded_model = AutoModel.from_pretrained(
            source,
            dtype=dtype,
            device_map=device,
            # sdpa avoids the flash-attn build; negligible cost at 1.7B.
            attn_implementation="sdpa",
            local_files_only=offline,
        )
        say("Loading processor…")
        loaded_processor = AutoProcessor.from_pretrained(
            source, fix_mistral_regex=True, local_files_only=offline
        )
        return loaded_model, loaded_processor

    loaded_offline = local_files_only
    try:
        model, processor = build(local_files_only)
    except Exception:
        # A cache can look complete and still be unusable (a truncated blob, a
        # revision mismatch). Fall back to the network unless told not to.
        if not local_files_only or forced:
            raise
        say("Local cache incomplete — retrying with the Hub…")
        model, processor = build(False)
        loaded_offline = False

    if not isinstance(model, Qwen3TTSForConditionalGeneration):
        raise TypeError(f"Expected Qwen3TTSForConditionalGeneration, got {type(model)}")
    model.eval()

    sample_rate = _resolve_sample_rate(model)

    languages = list(FALLBACK_LANGUAGES)
    try:
        reported = model.get_supported_languages()
        if reported:
            named = sorted({str(x).capitalize() for x in reported})
            languages = ["Auto"] + [x for x in named if x.lower() != "auto"]
    except Exception:
        pass

    say("Ready (loaded from local cache)." if loaded_offline else "Ready.")
    return LoadedModel(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        sample_rate=sample_rate,
        languages=languages,
        offline=loaded_offline,
    )


def _resolve_sample_rate(model) -> int:
    """Read the codec's output rate off the model rather than hardcoding it."""
    tok = getattr(model, "speech_tokenizer", None)
    if tok is not None:
        getter = getattr(tok, "get_output_sample_rate", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
        for attr in ("output_sample_rate", "sampling_rate", "sample_rate"):
            val = getattr(tok, attr, None)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
    return 24000


def missing_from_cache(model_id: str = MODEL_ID) -> list[str]:
    """Which required files are not in the local HF cache."""
    from huggingface_hub import try_to_load_from_cache

    missing = []
    for name in REQUIRED_FILES:
        try:
            hit = try_to_load_from_cache(model_id, name)
        except Exception:
            hit = None
        if not (isinstance(hit, str) and os.path.isfile(hit)):
            missing.append(name)
    return missing


def is_model_cached(model_id: str = MODEL_ID) -> bool:
    """True when the whole model is cached, so loading needs no network at all."""
    return not missing_from_cache(model_id)


def forced_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in ("1", "true", "yes")


def resolve_local_path(model_id: str = MODEL_ID) -> str | None:
    """The cached snapshot directory, or None if it isn't fully cached.

    Loading from a *path* rather than a repo id is what actually makes the load
    offline. Two places reach for the network even with `local_files_only=True`,
    and both are short-circuited by a local directory:

    * the tokenizer's Mistral-regex patch calls `model_info()` unconditionally,
      but only when `_is_local` is False;
    * the model's own `from_pretrained` fetches `speech_tokenizer/*` unless the
      path is already a directory.
    """
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None
