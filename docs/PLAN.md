# Implementation plan — next four features

Written after two measurements that change what's worth building. Both are
reproducible from the repo.

## Measured groundwork

**The description carries the voice, not the seed** ([seed_study.py](../scripts/seed_study.py)).
Across two different sentences: same description at different seeds gives timbre
similarity 0.929 and ΔF0 2.6 Hz; a different description gives 0.590 and 156 Hz.
Seed choice is ~2% of the effect of wording.

*Consequence:* multi-voice consistency is nearly free — reuse each character's
description. No per-character seed bookkeeping, no drift-correction machinery.

**`max_new_tokens` is not a real constraint.** At the 12 Hz codec, 300 characters
of text produces ~19 s of audio ≈ 228 codec frames — **5.6%** of a 4096 budget.
The chunker already caps chunks at 300 characters, so the ceiling is never
approached. My earlier concern about silent truncation was wrong.

*Consequence:* item 2 shrinks from a feature to a small correctness fix.

---

## 1. Sub-talker sampling — small, expert-facing

The model runs two sampling stages. The talker produces the prosodic/semantic
token stream; the **sub-talker** predicts the residual codebooks that carry
acoustic detail. `generate()` accepts `subtalker_dosample`, `subtalker_top_k`,
`subtalker_top_p`, `subtalker_temperature`, and the checkpoint's own
`generate_config` defaults them to `True / 50 / 1.0 / 0.9` — identical to the
talker defaults. The UI exposes none of them.

**Work**
- Add the four fields to `SynthRequest` (`engine/synth.py`), defaulted from the
  model's `generate_config` rather than hardcoded, and pass them to `generate()`.
- `ParamsPanel`: a collapsible **Sub-talker** section with a *Link to main*
  checkbox, checked by default so behaviour is unchanged until deliberately
  altered. Unchecking reveals the four controls.
- Persist in `Settings` and on `Take`, so restore-params and re-roll round-trip.

**Do this first:** sweep `subtalker_temperature` across 0.3–1.4 at a fixed talker
temperature and listen. Write what it actually does into the tooltip. Do not ship
a guess about the perceptual effect — if the sweep shows no audible difference,
say so in the tooltip and leave the control for completeness.

**Risk:** low. Additive, defaults preserve current behaviour.

## 2. max_new_tokens — correctness, not a feature

**Work**
- Default 4096 → **8192**, matching the checkpoint's `generate_config`; raise the
  spin range ceiling to match.
- Add truncation detection: if a generation returns a code sequence that hit the
  cap, mark the take and warn in the status bar. Cheap insurance that currently
  doesn't exist — today a truncated take is indistinguishable from a short one.
- Tooltip should state the real scale (~12 codec frames per second of audio) so
  the number is interpretable instead of magic.

**Risk:** very low. Half an hour of work.

## 3. Multi-voice dialogue — the big one

Cheaper than expected, because consistency comes free from stable descriptions.

**Script format.** Parse `SPEAKER: line` with continuation lines, blank-line
separation, and `[pause 1.5]` directives. Unlabelled leading text becomes the
default narrator. Keep it in `core/dialogue.py`, pure and unit-testable, with a
`DialogueLine(speaker, text, pause_after)` dataclass. Parse errors must point at
a line number, not fail silently.

**Cast.** A `Cast` maps speaker name → preset id, stored per project. UI: a cast
table (speaker, assigned voice, audition button) that auto-populates with the
speaker names found in the script and flags unassigned ones before generation
rather than failing mid-run.

**Generation.** Extend `SynthJob` with a `DIALOGUE` mode carrying a list of
`(line, instruct, seed)`. Reuse the existing per-item loop and progress/cancel
plumbing — this is the same shape as script chunking, with per-line descriptions
instead of one. Each character keeps its own seed only so a re-run reproduces the
same rendition; identity comes from the description.

**Assembly.** Concatenate with per-line pause control, defaulting to a slightly
longer gap on speaker change than within a speaker's own run.

**Output.** One joined take, plus optional per-line stems (numbered
`0007_VILLAIN.wav`) and a cue sheet of line → timestamp, which is what makes the
result usable in a video or game editor.

**Do this first:** confirm that two characters generated separately and
concatenated don't collide in level or room tone. If they do, normalisation moves
from "nice to have" to a requirement — see item 4's normalization section.

**Risk:** medium, mostly UI surface. The generation path is a variation on
existing code.

## 4. Pronunciation window — separate window, two halves

A non-modal `QDialog` (own window, closable independently), opened from the
script panel and from a menu.

### Dictionary half

Rules table: **enabled · match · replacement · whole-word · case-sensitive ·
regex · language scope · note**. Add/edit/delete/reorder, since order decides
precedence. Stored at `~/.local/share/voicestudio/pronunciation.json` via a
`core/pronounce.py` store mirroring `VoiceLibrary`.

Respelling is the only lever available — this model takes no phoneme input — so
the rules are plain text substitutions (`Qwen` → `Chwen`). Say that in the window
rather than implying IPA support.

**Tuning** is what makes it more than a find-and-replace table: select a rule,
hit **Test**, and it generates the phrase with and without the rule applied and
pins the two results into the existing A/B slots. This reuses the comparison
machinery already built and turns pronunciation fixing into the same
audition-and-compare loop as voice design.

### Normalization half

Checkboxes over a shared pipeline in `core/normalize.py`, each independently
toggleable: numbers to words, ordinals, currency, dates and times, unit
expansion, acronym spacing (`NASA` → `N A S A`), abbreviation expansion
(`Dr.` → `Doctor`), symbol replacement (`&` → `and`), quote/dash/whitespace
tidying, and bracketed-aside stripping.

**Order matters and must be fixed and visible:** normalize first, then user rules
last, so a hand-written rule can always override an automatic transform.

**Scope honestly.** Full number/date normalization is English-only in the first
pass. The other ten languages get whitespace and symbol tidying and a clearly
labelled "not yet localised" state. A half-working German number expander is
worse than none, because failures are silent and only audible at playback.

**Preview.** A live two-pane original → spoken-as view with changed spans
highlighted, plus a per-rule hit count so dead rules are visible.

**Integration.** Transformation happens on the text handed to the engine, in
`MainWindow._build_request` and the dialogue path. The script panel gains a
"spoken as" toggle so the user can see the transformed text without leaving the
editor. Store the transformed text on the `Take` alongside the original, so
history shows what was actually spoken.

**Risk:** medium. The normalizer is where the bugs will be; it needs its own unit
test file with a table of input → expected-output cases, which is also the
cheapest way to develop it.

---

## Suggested order

1. **max_new_tokens** — half an hour, removes a real if rare failure mode.
2. **Sub-talker** — small, and the sweep teaches us something about the model.
3. **Pronunciation window** — self-contained, no interaction with dialogue.
4. **Dialogue** — largest surface, and benefits from normalization existing.

Each lands independently; none blocks another.

## Verification

- `core/dialogue.py` and `core/normalize.py` get unit tests with input/expected
  tables — no model needed, so they run in milliseconds.
- Extend `scripts/gui_smoke.py`: cast assignment, a two-speaker dialogue producing
  the expected number of lines, and pronunciation rules altering the text actually
  sent to the engine.
- Extend `scripts/final_checks.py`: sub-talker params round-trip through a take,
  and truncation detection fires on a deliberately tiny `max_new_tokens`.
- Re-run `scripts/hidpi_check.py` — it builds every panel, so new UI must keep
  scaling and not clip.
