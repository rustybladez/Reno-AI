from datetime import date

from leave_assistant.config import DEFAULT_GEMINI_MODEL, load_settings, today_in_dhaka
from leave_assistant.schema import LLMOutput


def test_settings_defaults(monkeypatch):
    for name in ("LLM_PROVIDER", "GEMINI_MODEL", "POLICY_JSON", "POLICY_PATH"):
        monkeypatch.setenv(name, "")
    settings = load_settings()
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model == DEFAULT_GEMINI_MODEL
    assert settings.policy_json is None
    assert settings.policy_path.name == "policy.json"


def test_today_in_dhaka_is_a_date():
    assert isinstance(today_in_dhaka(), date)


def test_llm_output_parses_valid_json():
    raw = (
        '{"type": "not_covered", "answer": "", "citations": [], '
        '"uncovered_part": "", "calculations": [], "language": "en"}'
    )
    assert LLMOutput.model_validate_json(raw).type == "not_covered"
