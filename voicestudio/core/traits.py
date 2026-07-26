"""Trait vocabulary and the sentence composer behind the trait builder.

The builder assembles a natural-language voice description from dropdowns. The
composed text is always handed to the user as editable text — this is a starting
point, never a constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

ANY = "—"  # "unset" sentinel shown in every dropdown


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    options: tuple[str, ...]


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("age", "Age", (ANY, "child", "young", "adult", "middle-aged", "older", "elderly")),
    Dimension("subject", "Voice", (ANY, "man", "woman", "person")),
    Dimension("accent", "Accent", (
        ANY, "neutral", "American", "British", "Southern US", "Irish",
        "Scottish", "Australian", "Canadian", "Indian",
    )),
    Dimension("pitch", "Pitch", (ANY, "very low", "low", "mid", "high", "very high")),
    Dimension("timbre", "Timbre", (
        ANY, "warm", "bright", "deep", "smooth", "silky", "gravelly",
        "raspy", "breathy", "nasal", "metallic", "thin",
    )),
    Dimension("texture", "Texture", (
        ANY, "clean", "breathy", "gravelly", "raspy", "smoky", "metallic",
    )),
    Dimension("pace", "Pace", (ANY, "very slow", "slow", "measured", "brisk", "fast")),
    Dimension("energy", "Energy", (ANY, "calm", "relaxed", "moderate", "energetic", "explosive")),
    Dimension("emotion", "Emotion", (
        ANY, "neutral", "warm", "cheerful", "excited", "reverent", "soothing",
        "weary", "anxious", "menacing", "dramatic",
    )),
    Dimension("style", "Setting", (
        ANY, "documentary", "audiobook", "broadcast", "commercial",
        "intimate mic", "theatrical",
    )),
)

DIMENSION_BY_KEY = {d.key: d for d in DIMENSIONS}

_STYLE_CLAUSE = {
    "documentary": "as if narrating a nature documentary",
    "audiobook": "as if reading a novel aloud",
    "broadcast": "as if presenting on air",
    "commercial": "as if voicing an advertisement",
    "intimate mic": "speaking closely into the microphone",
    "theatrical": "with theatrical delivery",
}

_PACE_ADVERB = {
    "very slow": "very slowly",
    "slow": "slowly",
    "measured": "at a measured pace",
    "brisk": "briskly",
    "fast": "quickly",
}

_ENERGY_ADVERB = {
    "calm": "calmly",
    "relaxed": "in a relaxed way",
    "moderate": "evenly",
    "energetic": "energetically",
    "explosive": "explosively",
}

_EMOTION_CLAUSE = {
    "neutral": "an even, matter-of-fact tone",
    "warm": "a warm, friendly tone",
    "cheerful": "a bright, cheerful tone",
    "excited": "rising excitement",
    "reverent": "hushed reverence",
    "soothing": "a soothing, gentle tone",
    "weary": "a weary, world-worn tone",
    "anxious": "an anxious, unsteady tone",
    "menacing": "a cold, menacing edge",
    "dramatic": "heavy dramatic weight",
}


def _is_set(value: str | None) -> bool:
    return bool(value) and value != ANY


def _article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def compose(traits: dict[str, str]) -> str:
    """Turn a trait selection into an English voice description.

    Returns "" when nothing is selected, so the caller can leave the box alone.
    """
    get = lambda k: traits.get(k) if _is_set(traits.get(k)) else None  # noqa: E731

    age, subject, accent = get("age"), get("subject"), get("accent")
    pitch, timbre, texture = get("pitch"), get("timbre"), get("texture")
    pace, energy, emotion, style = get("pace"), get("energy"), get("emotion"), get("style")

    if not any([age, subject, accent, pitch, timbre, texture, pace, energy, emotion, style]):
        return ""

    # Subject phrase: "An older British man". Without an explicit subject the
    # noun is "voice", so the quality phrase must say "tone" to avoid repeating it.
    subject_noun = subject or "voice"
    quality_noun = "voice" if subject else "tone"
    subject_words = [w for w in (age, accent if accent != "neutral" else None, subject_noun) if w]
    sentence = f"{_article(subject_words[0])} {' '.join(subject_words)}"

    # Voice phrase: "with a low, gravelly, smoky voice"
    qualities = [q for q in (pitch, timbre, texture) if q]
    # Avoid "gravelly, gravelly" when timbre and texture agree.
    deduped: list[str] = []
    for q in qualities:
        if q not in deduped:
            deduped.append(q)
    if deduped:
        sentence += f" with a {', '.join(deduped)} {quality_noun}"
    if accent == "neutral":
        sentence += f"{' and' if deduped else ' with'} a neutral accent"

    # Delivery phrase: "speaking slowly and calmly"
    delivery = [adv for adv in (
        _PACE_ADVERB.get(pace) if pace else None,
        _ENERGY_ADVERB.get(energy) if energy else None,
    ) if adv]
    if delivery:
        sentence += f", speaking {' and '.join(delivery)}"

    if emotion:
        sentence += f", with {_EMOTION_CLAUSE.get(emotion, emotion)}"
    if style:
        sentence += f", {_STYLE_CLAUSE.get(style, style)}"

    return sentence.strip().rstrip(",") + "."


def empty_traits() -> dict[str, str]:
    return {d.key: ANY for d in DIMENSIONS}


def normalize(traits: dict | None) -> dict[str, str]:
    """Coerce a stored trait dict into a full, valid selection."""
    out = empty_traits()
    for key, value in (traits or {}).items():
        dim = DIMENSION_BY_KEY.get(key)
        if dim and value in dim.options:
            out[key] = value
    return out
