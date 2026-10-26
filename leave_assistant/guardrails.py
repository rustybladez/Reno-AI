"""Deterministic guardrail layer (PRD §9).

The model only classifies and drafts. This module decides what the user actually sees:
- the fallback and privacy texts are fixed templates, never generated;
- every citation must point at a real section AND quote text that is really in it;
- an answer with no verifiable citation becomes the fallback;
- every number in the answer is traced to the policy, a tool result or the user's input.
"""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError
from rapidfuzz import fuzz

from .policy import Policy
from .schema import AssistantResponse, Citation, LLMOutput

FALLBACK_TEXT = "I don't know, please contact HR."

PRIVACY_REFUSAL_TEXT = (
    "I'm sorry, but I can't share information about other employees. Leave records, including "
    "the reasons for leave and any medical information, are confidential and can only be accessed "
    "by the employee concerned, their line manager and authorised HR staff. If you have a question "
    "about a colleague's leave, please contact the HR Department"
)  # completed by privacy_refusal_text() with " at <HR email from the policy>."
# Fixed citations for the refusal (confidentiality clauses and the HR contact), so it shows a
# Source line and the policy text like any other answer. Chosen in code, never by the model.
PRIVACY_SOURCE_IDS = ["9.1", "9.2", "9.3", "11.2"]

SMALLTALK_TEXT = (
    "I'm here to help with the XYZ Company Leave Policy — for example annual, sick or casual "
    "leave, how to apply, or notice requirements."
)

PARTIAL_SUFFIX = "For the rest of your question: " + FALLBACK_TEXT

FUZZY_THRESHOLD = 90


# ---------- 1. parsing ----------

def parse_output(raw: str | None) -> LLMOutput | None:
    """Validate the model's raw text against the schema. None if invalid."""
    if not raw:
        return None
    text = raw.strip()
    # Tolerate a ```json fenced block.
    if fence := re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL):
        text = fence.group(1)
    try:
        return LLMOutput.model_validate_json(text)
    except ValidationError:
        return None


# ---------- 2. citation and quote verification ----------

def normalise(text: str) -> str:
    """Lowercase, unify quotes/dashes, drop table markup, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\|?(?:-{3,}\|)+", " ", text)  # markdown table separator rows
    text = text.replace("|", " ")
    text = " ".join(text.split())
    return text.strip(" .,;:!?")


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def quote_matches(quote: str, section_text: str) -> str | None:
    """Return "exact" or "fuzzy" if the quote is in the section text, else None.

    The quote may join fragments with "..." / "…"; each fragment must match.
    The fuzzy fallback only tolerates wording slips, never a changed number.
    """
    target = normalise(section_text)
    fragments = [normalise(f) for f in re.split(r"\.\.\.|…", quote)]
    fragments = [f for f in fragments if f]
    if not fragments:
        return None
    if all(f in target for f in fragments):
        return "exact"
    if all(
        fuzz.partial_ratio(f, target) >= FUZZY_THRESHOLD and _numbers(f) <= _numbers(target)
        for f in fragments
    ):
        return "fuzzy"
    return None


def clean_section_id(section_id: str) -> str:
    """'Section 4.1 (d)' / '§4.1(d)' / '[4.1(d)]' -> '4.1(d)'."""
    # The prompt shows ids in [brackets], so models sometimes copy the brackets.
    cleaned = section_id.strip().removeprefix("[").removesuffix("]").strip()
    cleaned = re.sub(r"^(section|sec\.?|§)\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace(" ", "")


@dataclass
class CitationCheck:
    section_id: str  # as given by the model
    quote: str
    resolved_id: str | None  # section the quote was found in (a child clause for a group id)
    method: str | None  # "exact" | "fuzzy" | None

    @property
    def valid(self) -> bool:
        return self.resolved_id is not None


def verify_citation(citation: Citation, policy: Policy) -> CitationCheck:
    section_id = clean_section_id(citation.section_id)
    if section := policy.get(section_id):
        candidates = [section]
    else:
        # A group id like "4.1" is valid if the quote is in one of its clauses.
        candidates = policy.descendants(section_id)
    for candidate in candidates:
        if method := quote_matches(citation.quote, candidate.text):
            return CitationCheck(citation.section_id, citation.quote, candidate.id, method)
    return CitationCheck(citation.section_id, citation.quote, None, None)


# ---------- 5. number provenance ----------

_SECTION_ID = r"\d+(?:\.\d+)?(?:\s?\([a-z]\))?"
# In a list ("Sections 4.1 and 4.3"), later ids must be dotted, so "Section 4.1 and 20 days"
# doesn't exempt the 20.
_LISTED_SECTION_ID = r"\d+\.\d+(?:\s?\([a-z]\))?"
_SECTION_REFERENCE = re.compile(
    rf"(?:\b(?:sections?|sec\.)\s*|§\s*){_SECTION_ID}(?:\s*(?:,|and|&|or)\s*§?\s*{_LISTED_SECTION_ID})*"
    rf"|\b\d+\.\d+\s?\([a-z]\)",  # "4.1(d)" is unambiguous even without "Section"
    flags=re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _to_number(token: str) -> float | None:
    token = token.replace(",", "").rstrip(".")
    try:
        return float(token)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[str]:
    return [m.group().rstrip(",.") for m in _NUMBER.finditer(text)]


def unverified_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in the answer that appear in none of the sources.

    Numbers inside a section reference ("Section 4.1", "4.1(d)", "§4.1") are skipped.
    A source number also matches when rounded to the answer's decimals (8333.333 -> 8,333.33).
    """
    answer_without_refs = _SECTION_REFERENCE.sub(" ", answer)
    source_values = {v for s in sources for t in extract_numbers(s) if (v := _to_number(t)) is not None}

    missing = []
    for token in extract_numbers(answer_without_refs):
        value = _to_number(token)
        if value is None:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        if not any(value == s or round(s, decimals) == value for s in source_values):
            missing.append(token)
    return missing


# ---------- 3. rendering ----------

def render_sources(section_ids: list[str], policy: Policy) -> str:
    labels = list(dict.fromkeys(policy.label(sid) for sid in section_ids))
    return "**Source:** " + "; ".join(labels)


def apply_guardrails(
    output: LLMOutput | None,
    policy: Policy,
    number_sources: list[str] | None = None,
) -> AssistantResponse:
    """Turn the model's (validated) output into the final response.

    number_sources: texts a number in the answer may legitimately come from besides the cited
    policy sections — the question, history, tool results and today's date.
    """
    debug: dict = {"model_type": output.type if output else None}

    if output is None:
        debug["reason"] = "invalid model output"
        return _fallback(debug)

    if output.type == "privacy_refusal":
        return _privacy_refusal(policy, debug)

    if output.type == "smalltalk":
        return AssistantResponse(type="smalltalk", text=SMALLTALK_TEXT, debug=debug)

    if output.type == "not_covered":
        return _fallback(debug)

    # answer / partial
    checks = [verify_citation(c, policy) for c in output.citations]
    debug["citation_checks"] = [
        {"section_id": c.section_id, "quote": c.quote, "valid": c.valid,
         "resolved_id": c.resolved_id, "method": c.method}
        for c in checks
    ]
    valid = [c for c in checks if c.valid]

    if not output.answer.strip():
        debug["reason"] = "empty answer"
        return _fallback(debug)
    if not valid:
        debug["reason"] = "no verifiable citation"
        return _fallback(debug)

    cited_ids = list(dict.fromkeys(c.resolved_id for c in valid))
    sources = [policy.by_id[sid].text for sid in cited_ids] + list(number_sources or [])
    debug["unverified_numbers"] = unverified_numbers(output.answer, sources)

    text = output.answer.strip() + "\n\n" + render_sources(cited_ids, policy)
    if output.type == "partial":
        text += "\n\n" + PARTIAL_SUFFIX
        debug["uncovered_part"] = output.uncovered_part

    return AssistantResponse(
        type=output.type,
        text=text,
        citations=[Citation(section_id=c.resolved_id, quote=c.quote) for c in valid],
        calculations=output.calculations,
        debug=debug,
    )


def privacy_refusal_text(policy: Policy) -> str:
    """The fixed refusal, ending with the HR email from the policy itself (not typed into the code)."""
    hr_email = policy.meta.get("hr_contact")
    return PRIVACY_REFUSAL_TEXT + (f" at {hr_email}." if hr_email else ".")


def _privacy_refusal(policy: Policy, debug: dict) -> AssistantResponse:
    """The fixed refusal text, then the Source line for the sections that exist in this policy."""
    ids = [sid for sid in PRIVACY_SOURCE_IDS if policy.get(sid)]
    text = privacy_refusal_text(policy) + ("\n\n" + render_sources(ids, policy) if ids else "")
    citations = [Citation(section_id=sid, quote=policy.by_id[sid].text) for sid in ids]
    return AssistantResponse(type="privacy_refusal", text=text, citations=citations, debug=debug)


def _fallback(debug: dict) -> AssistantResponse:
    return AssistantResponse(type="not_covered", text=FALLBACK_TEXT, debug=debug)


def tool_results_as_text(tool_log: list[dict]) -> list[str]:
    """Tool calls and results as plain strings, for number provenance."""
    return [json.dumps(entry, ensure_ascii=False) for entry in tool_log]
