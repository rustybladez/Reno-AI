"""Tries providers in order: if one is busy (rate limit, overload, timeout, no balance), the next
one answers the same request. Used to put DeepSeek and Gemini behind each other."""

from pydantic import BaseModel

from .base import Provider, ProviderBusyError, ProviderResult


class ChainProvider:
    def __init__(self, providers: list[Provider]):
        self.providers = providers
        self.current = providers[0]

    @property
    def name(self) -> str:
        return self.current.name

    @property
    def model(self) -> str:
        return self.current.model

    @property
    def models(self) -> list[str]:
        return [m for p in self.providers for m in getattr(p, "models", [p.model])]

    @property
    def thinking_level(self) -> str:
        return "; ".join(f"{p.name}: {getattr(p, 'thinking_level', None) or 'default'}" for p in self.providers)

    def backups(self) -> list[Provider]:
        """The other providers, for answering again when the current one gives unusable output."""
        return [p for p in self.providers if p is not self.current]

    def generate(
        self,
        system: str,
        messages: list[dict],
        schema: type[BaseModel],
        tools: list[dict] | None = None,
    ) -> ProviderResult:
        errors = []
        for provider in self.providers:
            try:
                result = provider.generate(system, messages, schema, tools)
                self.current = provider
                return result
            except ProviderBusyError as e:
                errors.append(f"{provider.name}: {e}")
        raise ProviderBusyError("All providers busy — " + " | ".join(errors))
