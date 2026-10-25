"""Data models shared by the assistant, guardrails, UI and evals."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ResponseType = Literal["answer", "partial", "not_covered", "privacy_refusal", "smalltalk"]


class Citation(BaseModel):
    section_id: str  # must match a section id in policy.json, e.g. "4.1(d)"
    quote: str  # verbatim excerpt from that section, <= 30 words


class LLMOutput(BaseModel):
    """What the model must return (also passed to the provider as the output schema).

    The model only classifies and drafts the answer. It never writes the fallback text
    or the privacy refusal; the guardrail layer renders those from fixed templates.
    """

    type: ResponseType
    answer: str  # required for answer/partial; "" otherwise
    citations: list[Citation]  # >= 1 for answer/partial; [] otherwise
    uncovered_part: str  # partial only; "" otherwise
    calculations: list[str]  # plain-language working for each tool result used
    language: Literal["en", "bn"]


class Turn(BaseModel):
    """One message in the chat history."""

    role: Literal["user", "assistant"]
    content: str


class AssistantResponse(BaseModel):
    """Final, rendered response returned by assistant.ask()."""

    type: ResponseType
    text: str  # what the user sees
    citations: list[Citation] = Field(default_factory=list)  # verified citations only
    calculations: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
