"""Split long scripts into synthesis-sized chunks."""

from __future__ import annotations

import re

# Sentence-ish boundaries, including CJK full-width punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])\s+|(?<=[.!?。！？…])(?=[^\s])")

# Every chunk boundary is an audible seam: each chunk is generated as an
# independent utterance, so it ends with a terminal pitch fall and the next one
# starts with a pitch reset. Measured, that doubles the pitch discontinuity at a
# sentence boundary (65 Hz across a join vs 32 Hz inside one pass).
#
# So the goal is *fewer chunks*, not smarter joining. Measured single-pass limits
# on this checkpoint: 2458 characters still renders in full (172 s of audio, 26%
# of the token budget); 3650 starts dropping text. 1500 keeps a wide margin while
# letting most scripts through in a single pass with no seams at all.
DEFAULT_MAX_CHARS = 1500
SAFE_SINGLE_PASS_CHARS = 2400


def split_script(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Break text on paragraph, then sentence, then word boundaries.

    Chunks stay under `max_chars` where possible so each generation is a
    reasonable length; a single unbroken sentence longer than the limit is
    split on whitespace rather than mid-word.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            chunks.extend(_split_paragraph(paragraph, max_chars))
    return chunks


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    out: list[str] = []
    buffer = ""
    for sentence in (s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()):
        if len(sentence) > max_chars:
            if buffer:
                out.append(buffer)
                buffer = ""
            out.extend(_split_on_words(sentence, max_chars))
            continue
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                out.append(buffer)
            buffer = sentence
    if buffer:
        out.append(buffer)
    return out


def _split_on_words(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split()
    if not words:
        return []

    out: list[str] = []
    buffer = ""
    for word in words:
        candidate = f"{buffer} {word}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            out.append(buffer)
        # A single word longer than the limit gets hard-sliced.
        while len(word) > max_chars:
            out.append(word[:max_chars])
            word = word[max_chars:]
        buffer = word
    if buffer:
        out.append(buffer)
    return out
