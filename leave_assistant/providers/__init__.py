"""LLM providers. All share the Provider protocol in base.py."""

from ..config import Settings
from .base import Provider, ProviderConfigError

PROVIDER_NAMES = ("openai_compat", "gemini")


def _gemini(settings: Settings, fallback: bool) -> Provider:
    from .gemini import GeminiProvider  # imported lazily so tests don't need the SDK set up

    models = [settings.gemini_model] + (settings.gemini_fallback_models if fallback else [])
    return GeminiProvider(settings.gemini_api_key, models, settings.gemini_thinking_level)


def _openai_compat(settings: Settings, fallback: bool) -> Provider:
    from .openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(
        settings.openai_compat_base_url,
        settings.openai_compat_api_key,
        settings.openai_compat_model,
        settings.openai_compat_thinking,
        settings.openai_compat_reasoning_effort,
    )


BUILDERS = {"openai_compat": _openai_compat, "gemini": _gemini}


def is_configured(name: str, settings: Settings) -> bool:
    if name == "gemini":
        return bool(settings.gemini_api_key)
    return bool(settings.openai_compat_base_url and settings.openai_compat_api_key and settings.openai_compat_model)


def get_provider(settings: Settings, fallback: bool = True) -> Provider:
    """LLM_PROVIDER first, then the other provider as a fallback — using whichever have keys.

    With only one key set, that provider is used on its own. fallback=False gives just the
    primary provider and model (evals use it to measure one model); it must be configured.
    """
    if settings.llm_provider not in BUILDERS:
        raise ProviderConfigError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}. Use 'openai_compat' or 'gemini'.")

    if not fallback:
        return BUILDERS[settings.llm_provider](settings, fallback=False)

    order = [settings.llm_provider] + [n for n in PROVIDER_NAMES if n != settings.llm_provider]
    providers = [BUILDERS[n](settings, fallback=True) for n in order if is_configured(n, settings)]
    if not providers:
        raise ProviderConfigError("No API key set. Set OPENAI_COMPAT_API_KEY (DeepSeek) and/or GEMINI_API_KEY.")
    if len(providers) == 1:
        return providers[0]

    from .chain import ChainProvider

    return ChainProvider(providers)
