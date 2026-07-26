"""Split long scripts into synthesis-sized chunks."""

from __future__ import annotations

import re

# Sentence-ish boundaries, including CJK full-width punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])\s+|(?<=[.!?。！？…])(?=[^\s])")
DEFAULT_MAX_CHARS = 300


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
