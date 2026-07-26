# Voice Design Studio

A desktop studio for **Qwen3-TTS-12Hz-1.7B-VoiceDesign** — the TTS model where you
*describe* a voice in plain language instead of picking one from a list.

Because the voice comes from a text description, the real work is iterative:
write a description, listen, change a word, compare against the last take. The app
is built around that loop rather than around a single "Generate" button.

![three-column layout: voice design, script and preview, takes](docs/screenshot.png)

## What's in it

**Voice design (left).** Twenty built-in templates across Narration, Informational,
Commercial, Character, and Stylized categories — pick one instead of facing an empty
box. A trait builder (age, accent, pitch, timbre, texture, pace, energy, emotion,
setting) composes an English sentence into the description field, which stays fully
editable; editing by hand detaches it from the traits rather than fighting them.
**Audition** generates any template against a fixed demo line and seed, so the picker
doubles as a browsable voice catalogue. Save anything you like into **My Voices** —
that forks a copy, so built-in templates are never modified.

**Script and preview (center).** The text to speak, a language selector (11 options
reported by the model), and sampling controls. Long scripts split on sentence
boundaries and generate chunk-by-chunk under one seed, then join into a single take.
The waveform is click-to-seek.

**Takes (right).** Every generation becomes a card with its seed and parameters.
Star the good ones, **re-roll** with a new seed, **restore** a take's settings back
into the editor, or export the WAV. Pin any two takes as **A** and **B** and toggle
between them — the fastest way to tell whether a description change actually helped.

**Seed lock** matters more than it sounds: lock the seed and a description edit
becomes the only variable between two takes. Unlock it, hit **Generate ×N**, and you
explore alternative readings of the same voice.

### What carries the voice

Worth knowing before you build a workflow on it, and measurable with
[seed_study.py](scripts/seed_study.py): across two *different* sentences,

| | timbre similarity | mean ΔF0 |
|---|---|---|
| same description, same seed | 0.935 | 6.1 Hz |
| same description, different seed | 0.929 | 2.6 Hz |
| different description | 0.590 | 156.3 Hz |

The **description** carries speaker identity — it stays recognisable at any seed.
The seed pins an exact *rendition*: identical text plus identical seed reproduces
bit-identical audio. So a character stays consistent across a script because its
description is stable, not because a seed is held fixed. Pinning a seed on a saved
voice gets that specific take back; it isn't what keeps the voice together.

## Setup

Requires a CUDA GPU (~6 GB VRAM) and roughly 5 GB of disk for weights.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

The first launch downloads ~4.5 GB from the Hugging Face Hub. Subsequent launches
load from cache in a few seconds.

Keyboard: `Ctrl+Enter` generate · `Space` play/pause · `Ctrl+S` save voice ·
`Ctrl+1`/`Ctrl+2` play A/B · `Esc` cancel.

## HiDPI

The UI follows your Qt display and font settings. Every size — text, padding,
button widths, the initial window — is derived from the application font
([metrics.py](voicestudio/ui/metrics.py)) rather than hardcoded pixels, so the
whole interface scales together instead of leaving small text in large panels.

That matters because Qt scales device-independent pixels by `devicePixelRatio`
but **not** by font DPI or your chosen system font size, which is how most
desktops actually scale text. Sizing off font metrics covers all three.

Fractional scale factors are preserved (`PassThrough` rounding), so 125% and
150% aren't rounded to 100% or 200%. The usual Qt knobs all work:

```bash
QT_FONT_DPI=144 ./run.sh          # scale text only
QT_SCALE_FACTOR=1.5 ./run.sh      # scale the whole UI
```

If text is still too small, the most direct fix is to raise your desktop's
interface font size — the app picks it up automatically. To preview a size
without changing your desktop:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/screenshot.py --pt 16 --out /tmp/preview.png
```

## How the model is loaded

`qwen3_tts` is **not** part of upstream transformers. The `qwen-tts` package ships
the transformers classes (`Qwen3TTSConfig`, `Qwen3TTSForConditionalGeneration`,
`Qwen3TTSProcessor`) and the app registers them into the transformers `Auto*`
mappings, then loads through the standard path:

```python
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
```

**Do not upgrade `transformers` past 4.57.3.** Version 5.x has no `qwen3_tts` and the
model will fail to load. `qwen-tts` pins this for you.

Generation is driven directly rather than through the `qwen_tts` convenience wrapper,
so the app controls seeding and per-request parameters. A voice description is sent as
a user turn and the spoken text as an assistant turn; the model returns codec tokens
that the bundled speech tokenizer decodes to 24 kHz mono audio. All of this lives in
[voicestudio/engine/](voicestudio/engine/) — the rest of the app never imports
transformers.

## Layout

```
voicestudio/
├── engine/    loader.py (registration + load)  synth.py (prompt, generate, decode)
├── core/      config  library  history  audio  script  templates  traits
├── assets/    templates.json — the 20 built-in voice descriptions
└── ui/        main_window  design_panel  script_panel  params_panel
               takes_panel  player  workers (QThread model host)  theme
```

Model load and generation run on a worker `QThread`; the UI thread only ever receives
signals, so the window stays responsive during the cold load and every generation.
Cancellation is checked between chunks and between variations.

Presets, take history, and session state persist under
`~/.local/share/voicestudio/`.

## Tests

```bash
.venv/bin/python scripts/smoke_test.py                               # engine only
.venv/bin/python scripts/final_checks.py                             # presets, cancel, VRAM
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/gui_smoke.py      # full app, headless
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/hidpi_check.py    # scaling, no model
```

`smoke_test.py` checks that audio is non-silent and that a fixed seed reproduces
identical output. `final_checks.py` covers preset forking, persistence,
cancellation, and VRAM stability across repeated generations, and confirms the
built-in templates actually produce distinct voices. `gui_smoke.py` drives the real
window: load, generate, variations with distinct seeds, A/B pinning, parameter
restore, and script chunking. `hidpi_check.py` rebuilds the panels at 9/12/16/22pt
and asserts nothing clips and that spacing scales with the font.
