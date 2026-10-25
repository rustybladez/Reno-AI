"""The interface every LLM provider implements.

Messages are provider-neutral dicts; each provider converts them to its own format:
    {"role": "user" | "assistant", "content": str}
    {"role": "assistant", "tool_calls": [ToolCall, ...]}
    {"role": "tool", "id": str | None, "name": str, "content": dict}   # a tool result
"""

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel


class ProviderBusyError(Exception):
    """Rate-limited or temporarily unavailable, after retries."""


class ProviderConfigError(Exception):
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
