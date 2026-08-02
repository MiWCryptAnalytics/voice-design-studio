"""Unit tests for the pure-logic modules. No model, no Qt — runs in milliseconds.

Run:  .venv/bin/python scripts/unit_tests.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicestudio.core import dialogue as dlg  # noqa: E402
from voicestudio.core.normalize import (  # noqa: E402
    NormalizationOptions,
    normalize,
    number_to_words,
    ordinal_to_words,
    year_to_words,
)
from voicestudio.core.pronounce import (  # noqa: E402
    ANY_LANGUAGE,
    PronunciationBook,
    PronunciationRule,
)
from voicestudio.core.script import split_script  # noqa: E402

failures: list[str] = []


def eq(label: str, got, want) -> None:
    ok = got == want
    print(f"[{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"       got:  {got!r}\n       want: {want!r}")
        failures.append(label)


def true(label: str, got, detail: str = "") -> None:
    ok = bool(got)
    print(f"[{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("\n--- number spelling ---")
for n, want in [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (20, "twenty"),
    (21, "twenty-one"), (99, "ninety-nine"), (100, "one hundred"),
    (105, "one hundred five"), (342, "three hundred forty-two"),
    (1000, "one thousand"), (1234, "one thousand two hundred thirty-four"),
    (1_000_000, "one million"), (-5, "minus five"),
]:
    eq(f"number_to_words({n})", number_to_words(n), want)

for n, want in [(1, "first"), (2, "second"), (3, "third"), (4, "fourth"),
                (5, "fifth"), (12, "twelfth"), (20, "twentieth"),
                (21, "twenty-first"), (103, "one hundred third")]:
    eq(f"ordinal_to_words({n})", ordinal_to_words(n), want)

for n, want in [(1999, "nineteen ninety-nine"), (2005, "two thousand five"),
                (2026, "twenty twenty-six"), (1900, "nineteen hundred"),
                (2010, "twenty ten")]:
    eq(f"year_to_words({n})", year_to_words(n), want)

print("\n--- normalization ---")
OPTS = NormalizationOptions()


def norm(text, language="English", **overrides):
    opts = NormalizationOptions(**{**OPTS.to_dict(), **overrides})
    return normalize(text, opts, language).text


eq("plain number", norm("I have 23 cats."), "I have twenty-three cats.")
eq("ordinal", norm("the 3rd time"), "the third time")
eq("currency", norm("It cost $5.50 today."),
   "It cost five dollars fifty cents today.")
eq("single dollar", norm("$1 only"), "one dollar only")
eq("units", norm("It is 5km away.", acronyms=False),
   "It is five kilometers away.")
eq("time on the hour", norm("at 3:00"), "at three o'clock")
eq("time with minutes", norm("at 3:45"), "at three forty-five")
eq("iso date", norm("on 2026-07-26 we go"),
   "on July twenty-sixth, twenty twenty-six we go")
eq("abbreviation", norm("Dr. Who", acronyms=False), "Doctor Who")
eq("symbol", norm("Tom & Jerry", acronyms=False), "Tom and Jerry")
eq("decimal", norm("pi is 3.14"), "pi is three point one four")
eq("initialism spelled out", norm("the FBI called"), "the F B I called")
eq("pronounceable acronym left alone", norm("NASA said"), "NASA said")
eq("brackets kept by default", norm("hello [aside] there", acronyms=False),
   "hello [aside] there")
eq("brackets stripped when asked",
   norm("hello [aside] there", acronyms=False, strip_brackets=True), "hello there")
eq("smart quotes tidied", norm("“hi”", acronyms=False), '"hi"')

eq("numeric steps skipped for other languages",
   norm("Ich habe 23 Katzen.", language="German", acronyms=False),
   "Ich habe 23 Katzen.")
eq("symbol tidying still runs for other languages",
   norm("Tom & Jerry", language="German", acronyms=False), "Tom and Jerry")

res = normalize("I paid $5 on 2026-07-26.", OPTS, "English")
true("result reports which steps fired", {"currency", "dates"} <= set(res.steps),
     str(res.steps))
true("no-op text reports no steps", not normalize("hello", OPTS, "English").steps)
eq("all_off is a true no-op",
   normalize("23 cats & $5", NormalizationOptions.all_off(), "English").text,
   "23 cats & $5")

print("\n--- pronunciation rules ---")
book = PronunciationBook(options=NormalizationOptions.all_off())
book.save = lambda: None  # keep the test off disk

r1 = book.add(PronunciationRule(match="Qwen", replacement="Chwen"))
eq("basic respelling", book.apply("Qwen is here", "English").text, "Chwen is here")
eq("case-insensitive by default", book.apply("qwen is here", "English").text,
   "Chwen is here")

book.add(PronunciationRule(match="cat", replacement="dog", whole_word=True))
eq("whole-word respects boundaries", book.apply("cat category", "English").text,
   "dog category")

book.add(PronunciationRule(match=r"\d+", replacement="N", regex=True))
eq("regex rule", book.apply("room 42", "English").text, "room N")

bad = PronunciationRule(match="([", replacement="x", regex=True)
true("invalid regex is caught, not raised", not bad.valid)
book.add(bad)
eq("invalid rule is skipped at synthesis", book.apply("([ stays", "English").text,
   "([ stays")

scoped = book.add(PronunciationRule(match="hello", replacement="bonjour",
                                    language="French"))
eq("language-scoped rule skipped elsewhere",
   book.apply("hello", "English").text, "hello")
eq("language-scoped rule fires in its language",
   book.apply("hello", "French").text, "bonjour")

hits = book.apply("Qwen and Qwen", "English")
eq("hit counts recorded", hits.rule_hits.get(r1.id), 2)
eq("skip_rule disables one rule",
   book.apply("Qwen", "English", skip_rule=r1.id).text, "Qwen")
eq("only_rule isolates one rule",
   book.apply("Qwen cat", "English", only_rule=r1.id).text, "Chwen cat")

order_book = PronunciationBook(options=NormalizationOptions.all_off())
order_book.save = lambda: None
first = order_book.add(PronunciationRule(match="a", replacement="b"))
order_book.add(PronunciationRule(match="b", replacement="c"))
eq("rules apply in order, each seeing the last output",
   order_book.apply("a", "English").text, "c")
order_book.move(first.id, +1)
eq("reordering changes the result", order_book.apply("a", "English").text, "b")

override = PronunciationBook(options=NormalizationOptions())
override.save = lambda: None
override.add(PronunciationRule(match="one hundred twenty-three",
                               replacement="a lot"))
eq("user rules run after normalization and can override it",
   override.apply("I counted 123 sheep", "English").text, "I counted a lot sheep")

disabled = PronunciationBook(options=NormalizationOptions(), enabled=False)
disabled.save = lambda: None
disabled.add(PronunciationRule(match="Qwen", replacement="Chwen"))
eq("book can be switched off entirely",
   disabled.apply("Qwen has 23", "English").text, "Qwen has 23")

print("\n--- dialogue parsing ---")
script = dlg.parse(
    "NARRATOR: The door opened.\n"
    "VILLAIN: You're late.\n"
    "  And I don't like waiting.\n"
    "[pause 1.5]\n"
    "NARRATOR: Nobody answered.\n"
)
eq("line count", len(script.lines), 3)
eq("speakers in appearance order", script.speakers, ["NARRATOR", "VILLAIN"])
eq("continuation folded into previous line", script.lines[1].text,
   "You're late. And I don't like waiting.")
eq("explicit pause attached to preceding line", script.lines[1].pause_after, 1.5)
eq("speaker-change pause applied", script.lines[0].pause_after,
   dlg.SPEAKER_CHANGE_PAUSE)
eq("last line has no trailing pause", script.lines[-1].pause_after, 0.0)
true("detected as dialogue", script.is_dialogue)

plain = dlg.parse("Just some prose.\nMore prose.")
eq("unlabelled text goes to the narrator", plain.speakers, [dlg.DEFAULT_SPEAKER])
true("plain prose is not dialogue", not plain.is_dialogue)

eq("prose containing a colon is not a cue",
   dlg.parse("He said this: it was over.").speakers, [dlg.DEFAULT_SPEAKER])
eq("speaker names normalise to upper case",
   dlg.parse("villain: hi").speakers, ["VILLAIN"])
eq("bare label then body",
   dlg.parse("VILLAIN:\n  Hello there.").lines[0].text, "Hello there.")
eq("empty script yields nothing", dlg.parse("   ").lines, [])

stray = dlg.parse("[pause 2]\nNARRATOR: hi")
true("pause before any line is reported", len(stray.issues) == 1)

cast_script = dlg.parse("A: one\nB: two")
eq("unassigned speakers detected",
   dlg.unassigned_speakers(cast_script, {"A": "preset-1"}), ["B"])
eq("no unassigned when cast complete",
   dlg.unassigned_speakers(cast_script, {"A": "p1", "B": "p2"}), [])

cue = dlg.format_cue_sheet(cast_script.lines, [1.5, 2.0], [0.6, 0.0])
true("cue sheet has a row per line", len(cue.splitlines()) == 3, cue.splitlines()[1])
true("cue sheet offsets accumulate", "00:02.100" in cue, cue)

print("\n--- utterance splitting ---")
import numpy as np  # noqa: E402

from voicestudio.core.segment import (  # noqa: E402
    find_silences,
    split_on_silence,
    trim_silence,
)

SR = 24000


def speech(seconds: float, freq: float = 140.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    # Amplitude-modulated tone stands in for voiced speech.
    return (0.5 * np.sin(2 * np.pi * freq * t)
            * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


three_parts = np.concatenate([
    speech(1.0), quiet(0.35), speech(1.0), quiet(0.35), speech(1.0)
])
found = find_silences(three_parts, SR)
true("finds the interior gaps", len(found) == 2, f"{len(found)} found")
true("gaps are reported longest first",
     len(found) < 2 or found[0][1] >= found[1][1])

parts = split_on_silence(three_parts, SR, ["aaaa", "bbbb", "cccc"])
true("splits into one segment per line", parts is not None and len(parts) == 3)
if parts:
    true("segments are all non-empty", all(p.size > 0 for p in parts))
    true("segments have similar length for similar text",
         max(p.size for p in parts) / min(p.size for p in parts) < 1.5)

eq("a single line needs no split",
   len(split_on_silence(speech(1.0), SR, ["only one"])), 1)
true("refuses when there aren't enough gaps",
     split_on_silence(speech(3.0), SR, ["a", "b", "c"]) is None)
true("refuses when the split contradicts the text proportions",
     split_on_silence(three_parts, SR,
                      ["tiny", "b" * 400, "tiny"]) is None)
true("empty text list yields nothing", split_on_silence(speech(1.0), SR, []) is None)

padded = np.concatenate([quiet(0.5), speech(1.0), quiet(0.5)])
trimmed = trim_silence(padded, SR)
true("leading and trailing silence trimmed", trimmed.size < padded.size * 0.75,
     f"{padded.size} -> {trimmed.size}")
true("all-silent audio survives trimming unchanged",
     trim_silence(quiet(0.5), SR).size == quiet(0.5).size)

from voicestudio.ui.workers import JobItem, _batches, _group_by_speaker  # noqa: E402

items = [
    JobItem(text="one", speaker="A"), JobItem(text="two", speaker="B"),
    JobItem(text="three", speaker="A"), JobItem(text="four", speaker="B"),
    JobItem(text="five", speaker="A"),
]
grouped = _group_by_speaker(items)
eq("groups one entry per speaker", [g[0] for g in grouped], ["A", "B"])
eq("group keeps original positions", [i for i, _ in grouped[0][1]], [0, 2, 4])
eq("speakers ordered by first appearance", grouped[1][0], "B")

long_items = [(i, JobItem(text="x" * 150, speaker="A")) for i in range(5)]
batches = _batches(long_items, 400)
true("long parts are split into passes", len(batches) > 1, f"{len(batches)} batches")
true("every line lands in exactly one batch",
     sum(len(b) for b in batches) == len(long_items))
eq("short parts stay in one pass", len(_batches(long_items[:2], 400)), 1)

print("\n--- script chunking (regression) ---")
from voicestudio.core.script import DEFAULT_MAX_CHARS  # noqa: E402

long_text = " ".join(f"Sentence number {i}." for i in range(40))
chunks = split_script(long_text, 120)
true("chunks stay under the limit", max(len(c) for c in chunks) <= 120)
eq("short text is one chunk", split_script("Hello."), ["Hello."])
# Fewer chunks means fewer seams; each join restarts the phrase contour.
true("default limit allows a whole page in one pass", DEFAULT_MAX_CHARS >= 1000,
     str(DEFAULT_MAX_CHARS))
eq("a 1000-char script needs no splitting", len(split_script("word " * 200)), 1)

print("\n--- crossfaded joining ---")
from voicestudio.core.audio import concat_crossfade  # noqa: E402

a, b = speech(1.0), speech(1.0, 180.0)
joined = concat_crossfade([a, b], SR, gap_seconds=0.0, fade_ms=15)
overlap = int(SR * 0.015)
true("crossfade shortens the result by the overlap",
     abs(joined.size - (a.size + b.size - overlap)) <= 2,
     f"{joined.size} vs {a.size + b.size - overlap}")
true("no hard discontinuity at the seam",
     float(np.abs(np.diff(joined)).max()) <= float(np.abs(np.diff(a)).max()) * 2.5)
eq("a single piece passes through", concat_crossfade([a], SR).size, a.size)
eq("nothing in, nothing out", concat_crossfade([], SR).size, 0)
gapped = concat_crossfade([a, b], SR, gap_seconds=0.25, fade_ms=15)
true("an explicit gap still lengthens the result", gapped.size > joined.size)

print("\n--- voice profiles ---")
import tempfile  # noqa: E402

from voicestudio.core.voiceprofile import VoiceProfile  # noqa: E402

profile = VoiceProfile(
    name="Warm narrator", embedding=np.arange(8, dtype=np.float64),
    source_wav="/tmp/ref.wav",
)
eq("embedding is normalized to float32", profile.embedding.dtype, np.dtype("float32"))
true("embedding is flattened to 1-D",
     VoiceProfile(name="x", embedding=np.ones((2, 4))).embedding.shape == (8,))

with tempfile.TemporaryDirectory() as tmp:
    saved = profile.save(Path(tmp) / "profile")  # suffix added automatically
    eq("save appends the .npz suffix", saved.suffix, ".npz")
    loaded = VoiceProfile.load(saved)
    eq("roundtrip keeps the name", loaded.name, profile.name)
    eq("roundtrip keeps the source path", loaded.source_wav, profile.source_wav)
    true("roundtrip keeps the embedding bit-exact",
         np.array_equal(loaded.embedding, profile.embedding))

try:
    VoiceProfile(name="empty", embedding=np.zeros(0))
    true("empty embedding is rejected", False)
except ValueError:
    true("empty embedding is rejected", True)

print("\n--- HiDPI auto-scale ---")
from voicestudio.ui.scaling import pick_scale  # noqa: E402  (pure, no Qt)

BARE = {}  # no overrides, unconfigured desktop
eq("27-inch 4K on bare X11 scales", pick_scale(96, 163, 1.0, BARE), 1.75)
eq("24-inch 4K rounds to whole step", pick_scale(96, 184, 1.0, BARE), 2.0)
eq("32-inch 4K gets a gentler step", pick_scale(96, 137, 1.0, BARE), 1.5)
eq("13-inch 4K laptop caps at 3", pick_scale(96, 331, 1.0, BARE), 3.0)
eq("1080p desktop stays unscaled", pick_scale(96, 92, 1.0, BARE), 1.0)
eq("4K TV at couch DPI stays unscaled", pick_scale(96, 80, 1.0, BARE), 1.0)
eq("implausible EDID (high) is ignored", pick_scale(96, 500, 1.0, BARE), 1.0)
eq("implausible EDID (low) is ignored", pick_scale(96, 20, 1.0, BARE), 1.0)
eq("compositor scaling wins", pick_scale(96, 163, 2.0, BARE), 1.0)
eq("configured Xft.dpi wins", pick_scale(144, 163, 1.0, BARE), 1.0)
eq("QT_SCALE_FACTOR wins",
   pick_scale(96, 163, 1.0, {"QT_SCALE_FACTOR": "2"}), 1.0)
eq("forced factor overrides everything",
   pick_scale(96, 163, 2.0, {"VOICESTUDIO_SCALE": "1.5"}), 1.5)
eq("forced 1 disables auto-scaling",
   pick_scale(96, 163, 1.0, {"VOICESTUDIO_SCALE": "1"}), 1.0)
eq("unparseable forced value falls back to auto",
   pick_scale(96, 163, 1.0, {"VOICESTUDIO_SCALE": "big"}), 1.75)
eq("forced factor clamps high",
   pick_scale(96, 96, 1.0, {"VOICESTUDIO_SCALE": "9"}), 3.0)
eq("forced factor clamps low",
   pick_scale(96, 96, 1.0, {"VOICESTUDIO_SCALE": "0.1"}), 0.5)

print("\nUNIT TESTS", "PASSED" if not failures else f"FAILED: {failures}")
raise SystemExit(1 if failures else 0)
