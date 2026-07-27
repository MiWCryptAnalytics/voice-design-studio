"""Synthesis: prompt construction, generation, and codec decoding.

This is the only module that touches the model's generation API. The prompt
format mirrors what Qwen3-TTS expects: the text to speak is framed as an
assistant turn, and the voice description is a preceding user turn.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import torch

from .loader import LoadedModel

MAX_SEED = 2**31 - 1


@dataclass
class SynthRequest:
    text: str
    instruct: str = ""
    language: str = "Auto"
    seed: int | None = None
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    # Matches the checkpoint's own generate_config. At ~12 codec frames per
    # second of audio this is far above any realistic chunk.
    max_new_tokens: int = 8192
    # Second sampling stage: predicts the residual codebooks (acoustic detail)
    # on top of the talker's stream. Defaults mirror the checkpoint.
    subtalker_do_sample: bool = True
    subtalker_temperature: float = 0.9
    subtalker_top_p: float = 1.0
    subtalker_top_k: int = 50

    def with_seed(self, seed: int) -> "SynthRequest":
        return replace(self, seed=seed)


@dataclass
class SynthResult:
    waveform: np.ndarray
    sample_rate: int
    seed: int
    elapsed: float
    request: SynthRequest
    codes: int = 0
    truncated: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.waveform) / float(self.sample_rate or 1)

    @property
    def realtime_factor(self) -> float:
        """Generated seconds of audio per second of compute. >1 is faster than realtime."""
        return self.duration / self.elapsed if self.elapsed > 0 else 0.0


def new_seed() -> int:
    return random.randint(0, MAX_SEED)


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assistant_prompt(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def _instruct_prompt(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


class SynthEngine:
    """Stateless-per-call wrapper around a LoadedModel."""

    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded

    @property
    def sample_rate(self) -> int:
        return self.loaded.sample_rate

    def _tokenize(self, text: str) -> torch.Tensor:
        enc = self.loaded.processor(text=text, return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(self.loaded.model.device)
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    def synthesize(self, req: SynthRequest) -> SynthResult:
        """Generate a single utterance."""
        results = self.synthesize_batch([req.text], req)
        return results[0]

    def synthesize_batch(
        self, texts: Sequence[str], req: SynthRequest
    ) -> list[SynthResult]:
        """Generate several chunks under one voice description and seed.

        Used for long scripts. Continuity across chunks comes from the shared
        *description*, not the seed — measurement shows seed choice barely moves
        speaker identity when the text differs. The shared seed only makes a
        re-run of the same script reproduce the same rendition.
        """
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            raise ValueError("Nothing to synthesize — the script is empty.")

        seed = req.seed if req.seed is not None else new_seed()
        _seed_everything(seed)

        model = self.loaded.model
        instruct = (req.instruct or "").strip()

        input_ids = [self._tokenize(_assistant_prompt(t)) for t in texts]
        instruct_ids = [
            self._tokenize(_instruct_prompt(instruct)) if instruct else None
            for _ in texts
        ]

        started = time.perf_counter()
        codes_list, _ = model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[req.language or "Auto"] * len(texts),
            non_streaming_mode=True,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
            subtalker_dosample=req.subtalker_do_sample,
            subtalker_temperature=req.subtalker_temperature,
            subtalker_top_p=req.subtalker_top_p,
            subtalker_top_k=req.subtalker_top_k,
        )
        wavs, fs = model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_list]
        )
        elapsed = time.perf_counter() - started

        out: list[SynthResult] = []
        for text, wav, codes in zip(texts, wavs, codes_list):
            n_codes = int(codes.shape[0])
            out.append(
                SynthResult(
                    waveform=_to_mono_float32(wav),
                    sample_rate=int(fs),
                    seed=seed,
                    # Attribute the batch cost evenly across chunks.
                    elapsed=elapsed / len(texts),
                    request=replace(req, text=text, seed=seed),
                    codes=n_codes,
                    truncated=_hit_token_cap(n_codes, req.max_new_tokens),
                )
            )
        return out


def _hit_token_cap(n_codes: int, max_new_tokens: int) -> bool:
    """True when generation stopped at the budget instead of at an end token.

    A run that ends naturally emits an EOS and returns well short of the cap;
    a truncated one returns `max_new_tokens - 1` codes (the final step produces
    no hidden state). Without this the audio just stops mid-word and the take
    looks like an ordinary short one.
    """
    return n_codes >= max_new_tokens - 1


def _to_mono_float32(wav) -> np.ndarray:
    arr = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        # Collapse any (channels, samples) / (samples, channels) layout to mono.
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    return np.ascontiguousarray(arr, dtype=np.float32)
