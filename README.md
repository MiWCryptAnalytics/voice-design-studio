# Voice Design Studio

A desktop studio for **Qwen3-TTS-12Hz-1.7B-VoiceDesign** — the TTS model where you
*describe* a voice in plain language instead of picking one from a list.

Because the voice comes from a text description, the real work is iterative:
write a description, listen, change a word, compare against the last take. The app
is built around that loop rather than around a single "Generate" button.

![three-column layout: voice design, script and preview, takes](docs/screenshot.png)

## Features

**Describe a voice, or start from a template.** Twenty built-in templates across
Narration, Informational, Commercial, Character, and Stylized categories. A trait
builder (age, accent, pitch, timbre, pace, energy, emotion…) composes a
description you can edit freely. **Audition** plays any template against a fixed
demo line, so the picker doubles as a browsable voice catalogue. Save keepers into
**My Voices**.

**Pin a voice exactly with a voice profile.** A description keeps a voice
*recognisable*; a profile keeps it *identical*. Drop or import a reference `.wav`,
or click **Extract Voice Profile** to lock the speaker of your current
description. From then on every take, script chunk, and re-roll is spoken by that
one fixed voice — across sessions too. Clone only voices you have the right to
use: your own recordings, consenting speakers, or samples generated from a
description.

**Write long scripts.** Up to 1500 characters renders in a single seamless pass.
Longer scripts split on sentence boundaries and are crossfaded back together —
set a voice profile first so every chunk keeps the same speaker.

**Multi-voice dialogue.** Write it like a screenplay and each character gets its
own voice:

```
NARRATOR: The door opened.
VILLAIN: You're late.
  And I don't like waiting.
[pause 1.5]
NARRATOR: Nobody answered.
```

Speaker labels are detected automatically and a **Cast** table appears for
assigning voices — your saved voices and the built-in templates, updated the
moment you save a new one. Indented lines continue the previous speaker and
`[pause N]` inserts silence. **Keep each voice steady across its lines** (on by
default) renders each character's part in one pass for consistency. **Tools →
Export dialogue stems** writes one WAV per line plus a cue sheet for a video or
game editor.

**Fix pronunciations** (`Ctrl+P`). A respelling dictionary (`Qwen` → `Chwen`)
with per-language, case, and regex options, plus automatic written-to-spoken
normalization: numbers, ordinals, currency, dates, times, units, initialisms.
A [live preview](docs/pronunciation.png) shows exactly what will be spoken, and
one click A/B-tests any rule on your own phrase.

**Compare everything.** Every generation becomes a take card with its seed and
parameters — star it, re-roll it with a new seed, restore its settings, or export
the WAV. Pin any two takes as **A** and **B** and toggle between them. Lock the
seed and a description edit becomes the only variable between takes; unlock it
and **Generate ×N** explores alternative readings.

## Getting started

You'll need Linux, Python 3.12+ (verified on 3.14), and a CUDA GPU — about 6 GB
of VRAM, or 10 GB if you use voice profiles (they keep a second checkpoint
loaded).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

The first launch downloads ~4.5 GB of model weights from the Hugging Face Hub;
the first voice-profile action downloads a further ~3.4 GB (the Base checkpoint,
which carries the speaker encoder). After that the app runs fully offline and
loads in about a second — set `VOICESTUDIO_OFFLINE=1` to forbid network access
outright.

Keyboard: `Ctrl+Enter` generate · `Space` play/pause · `Ctrl+S` save voice ·
`Ctrl+1`/`Ctrl+2` play A/B · `Esc` cancel.

On a HiDPI screen the app picks a sensible scale by itself when the desktop
hasn't configured one (a bare X11 session on a 4K monitor, say). Force a factor
with `VOICESTUDIO_SCALE=1.5`, disable with `VOICESTUDIO_SCALE=1`; any desktop
scaling and the usual `QT_FONT_DPI` / `QT_SCALE_FACTOR` knobs are respected
untouched.

## Tests

```bash
.venv/bin/python scripts/unit_tests.py                               # pure logic, no model
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/gui_smoke.py      # full app, headless
```

More suites (engine, offline, HiDPI) and six model-behaviour studies are
described in [DESIGN.md](DESIGN.md).

## Design notes

The defaults in this app are measured, not guessed — what actually carries a
voice (the description, not the seed), why chunk joins are audible (a pitch
reset *and* a speaker resample), why voice profiles exist, what the sub-talker
does, and how offline loading really works. All of it, with the study scripts
that reproduce the numbers, lives in [DESIGN.md](DESIGN.md).

## License

[GPL-3.0](LICENSE). Dependencies: [PyQt6](https://riverbankcomputing.com/software/pyqt/)
is GPL v3; the [`qwen-tts`](https://pypi.org/project/qwen-tts/) package and the
[Qwen3-TTS model weights](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
are Apache-2.0. The UI bundles the [IBM Plex Sans](https://github.com/IBM/plex)
typeface (SIL OFL 1.1 — license included alongside the font files).
