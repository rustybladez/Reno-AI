"""Settings loaded from environment variables (and a local .env file, if present).

Streamlit secrets are copied into the environment by app.py, so this package never
imports Streamlit.
"""

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMEZONE = ZoneInfo("Asia/Dhaka")

DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# Tried in order when the current model is busy or out of free quota (quotas are per model).
DEFAULT_GEMINI_FALLBACK_MODELS = "gemini-flash-lite-latest,gemini-3.1-flash-lite,gemini-3-flash-preview,gemini-3.5-flash"
DEFAULT_GEMINI_THINKING_LEVEL = "medium"  # at "low", lite models skipped tool calls and guessed dates (M4 evals)
DEFAULT_POLICY_PATH = "data/policy.json"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    gemini_api_key: str | None
    gemini_model: str
    gemini_fallback_models: list[str]
    gemini_thinking_level: str | None
    policy_json: str | None
    policy_path: Path
    openai_compat_base_url: str | None
    openai_compat_api_key: str | None
    openai_compat_model: str | None


def _env(name: str, default: str | None = None) -> str | None:
    """Read an env var, treating empty strings as unset."""
    value = os.environ.get(name, "").strip()
    return value or default


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    policy_path = Path(_env("POLICY_PATH", DEFAULT_POLICY_PATH))
    if not policy_path.is_absolute():
        policy_path = PROJECT_ROOT / policy_path

    return Settings(
        llm_provider=_env("LLM_PROVIDER", DEFAULT_PROVIDER).lower(),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        gemini_fallback_models=[
            m.strip() for m in _env("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACK_MODELS).split(",") if m.strip()
        ],
        gemini_thinking_level=_env("GEMINI_THINKING_LEVEL", DEFAULT_GEMINI_THINKING_LEVEL),
        policy_json=_env("POLICY_JSON"),
        policy_path=policy_path,
        openai_compat_base_url=_env("OPENAI_COMPAT_BASE_URL"),
        openai_compat_api_key=_env("OPENAI_COMPAT_API_KEY"),
        openai_compat_model=_env("OPENAI_COMPAT_MODEL"),
    )


def today_in_dhaka() -> date:
    return datetime.now(TIMEZONE).date()
