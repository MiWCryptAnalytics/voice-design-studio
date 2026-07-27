"""Pronunciation rules — user respellings applied on top of normalization.

This model takes no phoneme input, so the only lever is respelling the text
(`Qwen` -> `Chwen`). Rules are plain substitutions, applied in list order.

Pipeline order is fixed and deliberate: **normalize first, user rules last**, so
a hand-written rule can always override an automatic transform.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

from .config import DATA_DIR, ensure_dirs
from .normalize import NormalizationOptions, NormalizationResult, normalize

RULES_FILE = DATA_DIR / "pronunciation.json"
ANY_LANGUAGE = "Any"


@dataclass
class PronunciationRule:
    match: str
    replacement: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enabled: bool = True
    whole_word: bool = True
    case_sensitive: bool = False
    regex: bool = False
    language: str = ANY_LANGUAGE
    note: str = ""
    created: float = field(default_factory=time.time)

    def applies_to(self, language: str) -> bool:
        return (
            self.enabled
            and self.match != ""
            and self.language in (ANY_LANGUAGE, language)
        )

    def compile(self) -> re.Pattern | None:
        """None when the rule's own regex is invalid — never raises at synthesis."""
        pattern = self.match if self.regex else re.escape(self.match)
        if self.whole_word and not self.regex:
            pattern = rf"\b{pattern}\b"
        flags = 0 if self.case_sensitive else re.IGNORECASE
        try:
            return re.compile(pattern, flags)
        except re.error:
            return None

    @property
    def valid(self) -> bool:
        return self.compile() is not None

    def apply(self, text: str) -> tuple[str, int]:
        """Returns the rewritten text and how many times it fired."""
        pattern = self.compile()
        if pattern is None:
            return text, 0
        # Escape backslashes in the replacement so users type literal text.
        replacement = self.replacement.replace("\\", "\\\\")
        new_text, count = pattern.subn(replacement, text)
        return new_text, count


@dataclass
class SpokenText:
    """The result of turning written text into what will actually be spoken."""

    original: str
    text: str
    normalization_steps: list[str] = field(default_factory=list)
    rule_hits: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def total_rule_hits(self) -> int:
        return sum(self.rule_hits.values())


class PronunciationBook:
    """JSON-backed rule list plus the normalization settings."""

    def __init__(
        self,
        rules: list[PronunciationRule] | None = None,
        options: NormalizationOptions | None = None,
        enabled: bool = True,
    ):
        self._rules: list[PronunciationRule] = rules or []
        self.options = options or NormalizationOptions()
        self.enabled = enabled

    # ---------- persistence ----------

    @classmethod
    def load(cls) -> "PronunciationBook":
        try:
            raw = json.loads(RULES_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()

        known = set(PronunciationRule.__dataclass_fields__)
        rules = [
            PronunciationRule(**{k: v for k, v in entry.items() if k in known})
            for entry in raw.get("rules", [])
        ]
        opt_fields = set(NormalizationOptions.__dataclass_fields__)
        options = NormalizationOptions(
            **{k: bool(v) for k, v in raw.get("normalization", {}).items()
               if k in opt_fields}
        )
        return cls(rules, options, bool(raw.get("enabled", True)))

    def save(self) -> None:
        ensure_dirs()
        payload = {
            "version": 1,
            "enabled": self.enabled,
            "normalization": self.options.to_dict(),
            "rules": [asdict(r) for r in self._rules],
        }
        tmp = RULES_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(RULES_FILE)

    # ---------- rule management ----------

    def all(self) -> list[PronunciationRule]:
        return list(self._rules)

    def get(self, rule_id: str) -> PronunciationRule | None:
        return next((r for r in self._rules if r.id == rule_id), None)

    def add(self, rule: PronunciationRule) -> PronunciationRule:
        self._rules.append(rule)
        self.save()
        return rule

    def update(self, rule: PronunciationRule) -> None:
        for i, existing in enumerate(self._rules):
            if existing.id == rule.id:
                self._rules[i] = rule
                self.save()
                return

    def remove(self, rule_id: str) -> None:
        self._rules = [r for r in self._rules if r.id != rule_id]
        self.save()

    def move(self, rule_id: str, delta: int) -> None:
        """Reorder — earlier rules win, since each sees the previous one's output."""
        index = next((i for i, r in enumerate(self._rules) if r.id == rule_id), None)
        if index is None:
            return
        target = max(0, min(len(self._rules) - 1, index + delta))
        if target != index:
            self._rules.insert(target, self._rules.pop(index))
            self.save()

    # ---------- the pipeline ----------

    def apply(
        self,
        text: str,
        language: str = "English",
        use_normalization: bool = True,
        skip_rule: str | None = None,
        only_rule: str | None = None,
    ) -> SpokenText:
        """Normalize, then apply user rules in order.

        `skip_rule` / `only_rule` support the tuning A/B: generate the same line
        with and without a single rule.
        """
        original = text
        steps: list[str] = []

        if use_normalization and self.enabled:
            result: NormalizationResult = normalize(text, self.options, language)
            text, steps = result.text, result.steps

        hits: dict[str, int] = {}
        if self.enabled:
            for rule in self._rules:
                if rule.id == skip_rule:
                    continue
                if only_rule is not None and rule.id != only_rule:
                    continue
                if not rule.applies_to(language):
                    continue
                text, count = rule.apply(text)
                if count:
                    hits[rule.id] = count

        return SpokenText(
            original=original,
            text=text,
            normalization_steps=steps,
            rule_hits=hits,
        )
