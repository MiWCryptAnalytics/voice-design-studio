"""Text normalization: turn written forms into what should be spoken.

The model has no phoneme input, so everything here is plain text rewriting.

**Scope, stated honestly:** the numeric and date rules are English-only. For every
other language the numeric steps are skipped entirely rather than applied wrongly —
a half-working expander fails silently and is only audible at playback. Symbol and
whitespace tidying is language-agnostic and always available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

NUMERIC_LANGUAGES = {"english"}

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALES = [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"),
           (1_000_000, "million"), (1_000, "thousand")]

_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

_ABBREVIATIONS = {
    "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Dr": "Doctor",
    "Prof": "Professor", "St": "Saint", "Mt": "Mount", "Ave": "Avenue",
    "Blvd": "Boulevard", "Rd": "Road", "vs": "versus", "etc": "etcetera",
    "approx": "approximately", "Jr": "Junior", "Sr": "Senior",
}

_UNITS = {
    "km": "kilometers", "cm": "centimeters", "mm": "millimeters",
    "kg": "kilograms", "mg": "milligrams", "ml": "milliliters",
    "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes", "tb": "terabytes",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "mph": "miles per hour", "kph": "kilometers per hour",
    "ft": "feet", "lb": "pounds", "lbs": "pounds", "oz": "ounces",
}

_CURRENCIES = {"$": ("dollar", "cent"), "£": ("pound", "pence"),
               "€": ("euro", "cent"), "¥": ("yen", "sen")}

_SYMBOLS = {"&": " and ", "@": " at ", "%": " percent ", "+": " plus ",
            "=": " equals ", "#": " number ", "©": " copyright ",
            "°C": " degrees Celsius ", "°F": " degrees Fahrenheit "}


@dataclass(frozen=True)
class NormalizationOptions:
    """Each step is independently toggleable."""

    abbreviations: bool = True
    dates: bool = True
    times: bool = True
    currency: bool = True
    units: bool = True
    ordinals: bool = True
    numbers: bool = True
    acronyms: bool = True
    symbols: bool = True
    whitespace: bool = True
    strip_brackets: bool = False

    @classmethod
    def all_off(cls) -> "NormalizationOptions":
        return cls(**{f.name: False for f in fields(cls)})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class NormalizationResult:
    text: str
    steps: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.steps)


# ---------- number spelling ----------


def number_to_words(n: int) -> str:
    """Cardinal English for an integer."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rest]}" if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        out = f"{_ONES[hundreds]} hundred"
        return out + (f" {number_to_words(rest)}" if rest else "")
    for value, name in _SCALES:
        if n >= value:
            count, rest = divmod(n, value)
            out = f"{number_to_words(count)} {name}"
            return out + (f" {number_to_words(rest)}" if rest else "")
    return str(n)


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n)
    head, _, tail = words.rpartition(" ")
    last = tail or words
    stem, _, final = last.rpartition("-")

    def to_ordinal(word: str) -> str:
        if word in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[word]
        if word.endswith("y"):
            return word[:-1] + "ieth"
        return word + "th"

    converted = f"{stem}-{to_ordinal(final)}" if stem else to_ordinal(final)
    return f"{head} {converted}".strip()


def year_to_words(year: int) -> str:
    """Years as people actually say them.

    Paired for most centuries (1999 -> nineteen ninety-nine, 2026 -> twenty
    twenty-six), but 2000–2009 keeps the "two thousand five" form, which is the
    common reading for that decade.
    """
    if 2000 <= year <= 2009:
        return number_to_words(year)
    if 1100 <= year <= 2999:
        high, low = divmod(year, 100)
        if low == 0:
            return f"{number_to_words(high)} hundred"
        if low < 10:
            return f"{number_to_words(high)} oh {number_to_words(low)}"
        return f"{number_to_words(high)} {number_to_words(low)}"
    return number_to_words(year)


def _decimal_to_words(whole: str, frac: str) -> str:
    digits = " ".join(_ONES[int(d)] for d in frac)
    return f"{number_to_words(int(whole))} point {digits}"


# ---------- individual steps ----------


def _step_strip_brackets(text: str) -> str:
    return re.sub(r"[\[(]{1}[^\])]*[\])]{1}", " ", text)


def _step_abbreviations(text: str) -> str:
    for short, long in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\.?(?=\s|$)", long, text)
    return text


def _step_dates(text: str) -> str:
    def iso(match: re.Match) -> str:
        year, month, day = (int(g) for g in match.groups())
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return match.group(0)
        return f"{_MONTHS[month - 1]} {ordinal_to_words(day)}, {year_to_words(year)}"

    def us(match: re.Match) -> str:
        month, day, year = (int(g) for g in match.groups())
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return match.group(0)
        return f"{_MONTHS[month - 1]} {ordinal_to_words(day)}, {year_to_words(year)}"

    text = re.sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", iso, text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", us, text)
    return text


def _step_times(text: str) -> str:
    def repl(match: re.Match) -> str:
        hour, minute = int(match.group(1)), int(match.group(2))
        suffix = match.group(3)
        if hour > 23 or minute > 59:
            return match.group(0)
        spoken = number_to_words(hour)
        if minute == 0:
            spoken += " o'clock" if not suffix else ""
        elif minute < 10:
            spoken += f" oh {number_to_words(minute)}"
        else:
            spoken += f" {number_to_words(minute)}"
        if suffix:
            spoken += " " + " ".join(suffix.replace(".", "").lower())
        return spoken

    return re.sub(r"\b(\d{1,2}):(\d{2})\s*([ap]\.?m\.?)?", repl, text,
                  flags=re.IGNORECASE)


def _step_currency(text: str) -> str:
    symbols = "".join(re.escape(s) for s in _CURRENCIES)

    def repl(match: re.Match) -> str:
        symbol, whole, frac = match.group(1), match.group(2), match.group(3)
        major, minor = _CURRENCIES[symbol]
        amount = int(whole.replace(",", ""))
        out = f"{number_to_words(amount)} {major}{'' if amount == 1 else 's'}"
        if frac and int(frac):
            cents = int(frac.ljust(2, "0")[:2])
            out += f" {number_to_words(cents)} {minor}{'' if cents == 1 else 's'}"
        return out

    return re.sub(rf"([{symbols}])(\d[\d,]*)(?:\.(\d{{1,2}}))?", repl, text)


def _step_units(text: str) -> str:
    names = sorted(_UNITS, key=len, reverse=True)
    pattern = rf"\b(\d+(?:\.\d+)?)\s?({'|'.join(names)})\b"

    def repl(match: re.Match) -> str:
        return f"{match.group(1)} {_UNITS[match.group(2).lower()]}"

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def _step_ordinals(text: str) -> str:
    return re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        lambda m: ordinal_to_words(int(m.group(1))),
        text,
        flags=re.IGNORECASE,
    )


def _step_numbers(text: str) -> str:
    text = re.sub(
        r"\b(\d+)\.(\d+)\b",
        lambda m: _decimal_to_words(m.group(1), m.group(2)),
        text,
    )
    return re.sub(
        r"\b\d[\d,]*\b",
        lambda m: number_to_words(int(m.group(0).replace(",", ""))),
        text,
    )


def _step_acronyms(text: str) -> str:
    """Space out initialisms, but leave pronounceable acronyms alone.

    Heuristic: 2–5 uppercase letters are spelled out only when they have no
    vowel (FBI, CPU) or are very short (UI). NASA and SONAR keep their word
    form, which is how people actually say them.
    """

    def repl(match: re.Match) -> str:
        word = match.group(0)
        has_vowel = any(c in "AEIOU" for c in word)
        if has_vowel and len(word) > 3:
            return word
        return " ".join(word)

    return re.sub(r"\b[A-Z]{2,5}\b", repl, text)


def _step_symbols(text: str) -> str:
    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)
    return text


def _step_whitespace(text: str) -> str:
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# Order is fixed and meaningful: specific patterns claim their text before
# general ones (currency and dates before bare numbers), tidying runs last.
_PIPELINE = [
    ("strip_brackets", _step_strip_brackets, False),
    ("abbreviations", _step_abbreviations, False),
    ("dates", _step_dates, True),
    ("times", _step_times, True),
    ("currency", _step_currency, True),
    ("units", _step_units, True),
    ("ordinals", _step_ordinals, True),
    ("numbers", _step_numbers, True),
    ("acronyms", _step_acronyms, False),
    ("symbols", _step_symbols, False),
    ("whitespace", _step_whitespace, False),
]

STEP_LABELS = {
    "strip_brackets": "Remove [bracketed] asides",
    "abbreviations": "Expand abbreviations (Dr. → Doctor)",
    "dates": "Speak dates (2026-07-26 → July twenty-sixth…)",
    "times": "Speak times (3:45 → three forty-five)",
    "currency": "Speak currency ($5.50 → five dollars fifty cents)",
    "units": "Expand units (5km → five kilometers)",
    "ordinals": "Speak ordinals (3rd → third)",
    "numbers": "Speak numbers (123 → one hundred twenty-three)",
    "acronyms": "Spell out initialisms (FBI → F B I)",
    "symbols": "Replace symbols (& → and)",
    "whitespace": "Tidy quotes, dashes and spacing",
}

NUMERIC_STEPS = {name for name, _, numeric in _PIPELINE if numeric}


def supports_numeric(language: str) -> bool:
    return (language or "").strip().lower() in NUMERIC_LANGUAGES


def normalize(
    text: str,
    options: NormalizationOptions | None = None,
    language: str = "English",
) -> NormalizationResult:
    """Apply the enabled steps in fixed order.

    Numeric steps are skipped for languages we have not localised, so text
    passes through untouched rather than being mangled.
    """
    options = options or NormalizationOptions()
    numeric_ok = supports_numeric(language)
    applied: list[str] = []

    for name, func, is_numeric in _PIPELINE:
        if not getattr(options, name):
            continue
        if is_numeric and not numeric_ok:
            continue
        before = text
        text = func(text)
        if text != before:
            applied.append(name)

    return NormalizationResult(text=text, steps=applied)
