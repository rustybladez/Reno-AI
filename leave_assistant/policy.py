"""Load the hand-verified policy and look up its sections."""

import json
from dataclasses import dataclass

from .config import Settings

POLICY_NOT_FOUND_MESSAGE = "Policy file not found. See README → Setup."


class PolicyNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class Section:
    id: str  # "4.1(d)", "2.2", "7", or "header"
    parent: str  # "4.1", "2", or ""
    heading: str  # "Annual Leave"
    path: str  # "4. Types of Leave > 4.1 Annual Leave"
    text: str  # verbatim policy text


class Policy:
    def __init__(self, meta: dict, sections: list[Section]):
        self.meta = meta
        self.sections = sections
        self.by_id = {s.id: s for s in sections}

    def get(self, section_id: str) -> Section | None:
        return self.by_id.get(section_id)

    def descendants(self, section_id: str) -> list[Section]:
        """Clauses under a group id: "4.1" -> 4.1(a)..(e); "4" -> 4.0, 4.1(a), ..."""
        return [
            s for s in self.sections
            if s.id.startswith(section_id + "(") or s.id.startswith(section_id + ".")
        ]

    def heading_for(self, section_id: str) -> str:
        """Heading of an entry, or of a group id that only appears in paths ("4.1", "4")."""
        if section := self.get(section_id):
            return section.heading
        for s in self.descendants(section_id):
            for part in s.path.split(" > "):
                number, _, title = part.partition(" ")
                if number.rstrip(".") == section_id:
                    return title
        return ""

    def label(self, section_id: str) -> str:
        """Citation label, e.g. "Section 4.1(d) – Annual Leave"."""
        if section_id == "header":
            return f"Policy details ({self.meta.get('policy_number', '')})".replace(" ()", "")
        heading = self.heading_for(section_id)
        return f"Section {section_id} – {heading}" if heading else f"Section {section_id}"

    def render_for_prompt(self) -> str:
        """One block per section: "[4.1(d)] (4.1 Annual Leave) Up to 10 unused days ..." """
        blocks = []
        for s in self.sections:
            location = s.path.split(" > ")[-1]
            blocks.append(f"[{s.id}] ({location}) {s.text}")
        return "\n\n".join(blocks)


def parse_policy(data: dict) -> Policy:
    sections = [
        Section(
            id=s["id"],
            parent=s.get("parent", ""),
            heading=s["heading"],
            path=s.get("path", s["heading"]),
            text=s["text"],
        )
        for s in data["sections"]
    ]
    return Policy(meta=data.get("meta", {}), sections=sections)


def load_policy(settings: Settings) -> Policy:
    """POLICY_JSON (secret/env) first, then the file at POLICY_PATH. No fallback to the example."""
    if settings.policy_json:
        return parse_policy(json.loads(settings.policy_json))
    if settings.policy_path.exists():
        return parse_policy(json.loads(settings.policy_path.read_text(encoding="utf-8")))
    raise PolicyNotFoundError(POLICY_NOT_FOUND_MESSAGE)
