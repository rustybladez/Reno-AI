"""The interface every LLM provider implements.

Messages are provider-neutral dicts; each provider converts them to its own format:
    {"role": "user" | "assistant", "content": str}
    {"role": "assistant", "tool_calls": [ToolCall, ...], "raw": <provider's own turn or None>}
    {"role": "tool", "id": str | None, "name": str, "content": dict}   # a tool result
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


class ProviderError(Exception):
    """The provider call failed."""


class ProviderBusyError(ProviderError):
    """Rate-limited or temporarily unavailable, after retries."""


class ProviderConfigError(ProviderError):
    """Missing API key or unknown provider."""


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class ProviderResult:
    """Either the final text (JSON) or tool calls the model wants to make."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The provider's own record of the model turn, replayed as-is on the next request
    # (Gemini needs its "thought signatures" sent back with function calls).
    raw: Any = None
    model: str | None = None  # which model produced this (providers may switch models)


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self,
        system: str,
        messages: list[dict],
        schema: type[BaseModel],
        tools: list[dict] | None = None,
    ) -> ProviderResult: ...
