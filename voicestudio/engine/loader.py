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

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

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
) -> LoadedModel:
    """Load model + processor. `progress` receives human-readable status strings."""

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    say("Registering qwen3_tts with transformers…")
    _register_once()

    device = device or pick_device()
    dtype = pick_dtype(device)

    say(f"Loading weights onto {device} ({str(dtype).replace('torch.', '')})…")
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        # sdpa avoids the flash-attn build; negligible cost at 1.7B.
        attn_implementation="sdpa",
    )
    if not isinstance(model, Qwen3TTSForConditionalGeneration):
        raise TypeError(f"Expected Qwen3TTSForConditionalGeneration, got {type(model)}")
    model.eval()

    say("Loading processor…")
    processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)

    sample_rate = _resolve_sample_rate(model)

    languages = list(FALLBACK_LANGUAGES)
    try:
        reported = model.get_supported_languages()
        if reported:
            named = sorted({str(x).capitalize() for x in reported})
            languages = ["Auto"] + [x for x in named if x.lower() != "auto"]
    except Exception:
        pass

    say("Ready.")
    return LoadedModel(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        sample_rate=sample_rate,
        languages=languages,
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


def is_model_cached(model_id: str = MODEL_ID) -> bool:
    """True if weights are already in the HF cache (so load won't hit the network)."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(model_id, "config.json")
    return isinstance(hit, str) and os.path.isfile(hit)
