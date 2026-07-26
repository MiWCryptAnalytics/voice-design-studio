"""Built-in voice description templates (read-only starters)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from .config import TEMPLATES_FILE

DEFAULT_DEMO_SENTENCE = "The quiet between two thoughts is where the whole story hides."


@dataclass(frozen=True)
class VoiceTemplate:
    id: str
    name: str
    category: str
    instruct: str
    tags: tuple[str, ...] = ()
    traits: dict = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        haystack = " ".join(
            [self.name, self.category, self.instruct, " ".join(self.tags)]
        ).lower()
        return all(term in haystack for term in q.split())


@dataclass(frozen=True)
class TemplateSet:
    templates: tuple[VoiceTemplate, ...]
    demo_sentence: str = DEFAULT_DEMO_SENTENCE
    demo_seed: int = 20260725

    def categories(self) -> list[str]:
        seen: list[str] = []
        for t in self.templates:
            if t.category not in seen:
                seen.append(t.category)
        return seen

    def by_category(self, query: str = "") -> dict[str, list[VoiceTemplate]]:
        grouped: dict[str, list[VoiceTemplate]] = {}
        for t in self.templates:
            if t.matches(query):
                grouped.setdefault(t.category, []).append(t)
        return grouped

    def get(self, template_id: str) -> VoiceTemplate | None:
        return next((t for t in self.templates if t.id == template_id), None)


@lru_cache(maxsize=1)
def load_templates() -> TemplateSet:
    try:
        raw = json.loads(TEMPLATES_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return TemplateSet(templates=())

    items = []
    for entry in raw.get("templates", []):
        items.append(
            VoiceTemplate(
                id=entry.get("id", ""),
                name=entry.get("name", "Untitled"),
                category=entry.get("category", "Other"),
                instruct=entry.get("instruct", ""),
                tags=tuple(entry.get("tags", [])),
                traits=entry.get("traits", {}) or {},
            )
        )
    return TemplateSet(
        templates=tuple(items),
        demo_sentence=raw.get("demo_sentence", DEFAULT_DEMO_SENTENCE),
        demo_seed=int(raw.get("demo_seed", 20260725)),
    )
