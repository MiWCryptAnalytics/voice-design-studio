"""Parse screenplay-style multi-voice scripts.

    NARRATOR: The door opened.
    VILLAIN: You're late.
      And I don't like waiting.
    [pause 1.5]
    NARRATOR: Nobody answered.

Rules:
- `SPEAKER: text` starts a line for that speaker.
- An indented or unlabelled following line continues the previous speaker.
- `[pause N]` on its own line inserts N seconds of silence.
- Text before any speaker label belongs to the default speaker (the narrator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_SPEAKER = "NARRATOR"
DEFAULT_PAUSE = 0.35
SPEAKER_CHANGE_PAUSE = 0.6

# A label is a short run of letters/digits/spaces before a colon. Bounded length
# stops ordinary prose containing a colon from being mistaken for a cue.
_SPEAKER_RE = re.compile(r"^\s{0,3}([A-Za-z][A-Za-z0-9 ._'-]{0,31}):\s*(.*)$")
_PAUSE_RE = re.compile(r"^\s*\[\s*pause\s+(\d+(?:\.\d+)?)\s*\]\s*$", re.IGNORECASE)


def _is_speaker_label(label: str) -> bool:
    """Distinguish a cue from ordinary prose that happens to contain a colon.

    "He said this: it was over." must not become a speaker called HE SAID THIS.
    A cue is a single word, or ALL CAPS, or Title Case across at most 3 words.
    """
    words = label.split()
    if not words or len(words) > 3:
        return False
    if len(words) == 1:
        return True
    if label.upper() == label:
        return True
    return all(w[:1].isupper() for w in words)


@dataclass
class DialogueLine:
    speaker: str
    text: str
    pause_after: float = DEFAULT_PAUSE
    line_number: int = 0


@dataclass
class ParseIssue:
    line_number: int
    message: str


@dataclass
class DialogueScript:
    lines: list[DialogueLine] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        """Distinct speakers in first-appearance order."""
        seen: list[str] = []
        for line in self.lines:
            if line.speaker not in seen:
                seen.append(line.speaker)
        return seen

    @property
    def is_dialogue(self) -> bool:
        """True when the script actually uses speaker labels."""
        return any(line.speaker != DEFAULT_SPEAKER for line in self.lines) or (
            len(self.speakers) > 1
        )

    def total_characters(self) -> int:
        return sum(len(line.text) for line in self.lines)


def parse(text: str, default_speaker: str = DEFAULT_SPEAKER) -> DialogueScript:
    script = DialogueScript()
    current: DialogueLine | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.text.strip():
            current.text = " ".join(current.text.split())
            script.lines.append(current)
        current = None

    for number, raw in enumerate((text or "").splitlines(), start=1):
        stripped = raw.strip()

        if not stripped:
            flush()
            continue

        pause = _PAUSE_RE.match(raw)
        if pause:
            seconds = float(pause.group(1))
            flush()
            if script.lines:
                script.lines[-1].pause_after = seconds
            else:
                script.issues.append(
                    ParseIssue(number, "Pause before any spoken line — ignored.")
                )
            continue

        match = _SPEAKER_RE.match(raw)
        if match and _is_speaker_label(match.group(1)):
            speaker = " ".join(match.group(1).split()).upper()
            body = match.group(2).strip()
            flush()
            current = DialogueLine(speaker=speaker, text=body, line_number=number)
            if not body:
                # A bare "SPEAKER:" is fine — the next lines continue it.
                continue
            continue

        if current is None:
            current = DialogueLine(
                speaker=default_speaker, text=stripped, line_number=number
            )
        else:
            current.text += " " + stripped

    flush()
    _apply_speaker_change_pauses(script)
    return script


def _apply_speaker_change_pauses(script: DialogueScript) -> None:
    """Leave a longer beat when the speaker changes, as in real conversation."""
    for i, line in enumerate(script.lines[:-1]):
        if line.pause_after != DEFAULT_PAUSE:
            continue  # an explicit [pause] wins
        if script.lines[i + 1].speaker != line.speaker:
            line.pause_after = SPEAKER_CHANGE_PAUSE
    if script.lines:
        script.lines[-1].pause_after = 0.0


def unassigned_speakers(script: DialogueScript, cast: dict[str, str]) -> list[str]:
    """Speakers with no voice assigned — checked before generating, not during."""
    return [s for s in script.speakers if not cast.get(s)]


def format_cue_sheet(lines: list[DialogueLine], durations: list[float],
                     gaps: list[float]) -> str:
    """A `timestamp  SPEAKER  text` listing for use in an editor."""
    out = ["# start\tend\tspeaker\ttext"]
    cursor = 0.0
    for line, duration, gap in zip(lines, durations, gaps):
        end = cursor + duration
        out.append(
            f"{_timestamp(cursor)}\t{_timestamp(end)}\t{line.speaker}\t{line.text}"
        )
        cursor = end + gap
    return "\n".join(out)


def _timestamp(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes):02d}:{rest:06.3f}"
