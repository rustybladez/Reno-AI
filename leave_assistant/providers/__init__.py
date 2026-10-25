"""LLM providers. All share the Provider protocol in base.py."""

from ..config import Settings
from .base import Provider, ProviderConfigError


def get_provider(settings: Settings) -> Provider:
    if settings.llm_provider == "gemini":
        raise ProviderConfigError("The Gemini provider is added in milestone M3.")
    raise ProviderConfigError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Use 'gemini'.")
