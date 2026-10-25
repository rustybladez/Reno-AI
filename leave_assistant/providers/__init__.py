"""LLM providers. All share the Provider protocol in base.py."""

from ..config import Settings
from .base import Provider, ProviderConfigError


def get_provider(settings: Settings) -> Provider:
    if settings.llm_provider == "gemini":
        from .gemini import GeminiProvider  # imported lazily so tests don't need the SDK set up

        models = [settings.gemini_model] + settings.gemini_fallback_models
        return GeminiProvider(settings.gemini_api_key, models, settings.gemini_thinking_level)
    raise ProviderConfigError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Use 'gemini'.")
