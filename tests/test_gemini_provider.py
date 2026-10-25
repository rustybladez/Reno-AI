"""Offline tests for the Gemini provider: conversions and model cycling (no network)."""

import pytest
from google.genai import errors, types

from leave_assistant.providers import gemini
from leave_assistant.providers.base import ProviderBusyError, ProviderConfigError, ProviderError, ToolCall
from leave_assistant.providers.gemini import (
    PLACEHOLDER_SIGNATURE,
    GeminiProvider,
    GeminiTurn,
    _cooldown,
    _to_contents,
    _to_gemini_tools,
)
from leave_assistant.tools import TOOL_SPECS

QUESTION = [{"role": "user", "content": "q"}]


def test_missing_key_is_a_config_error():
    with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY"):
        GeminiProvider(None, ["some-model"])


# ---------- conversions ----------

def _messages_with_tool_round(raw):
    return [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "tool_calls": [ToolCall("calculate", {}, id="a"), ToolCall("weekday_of", {}, id="b")], "raw": raw},
        {"role": "tool", "id": "a", "name": "calculate", "content": {"result": 1}},
        {"role": "tool", "id": "b", "name": "weekday_of", "content": {"weekday": "Monday"}},
    ]


def test_same_model_replays_its_own_turn_and_groups_tool_results():
    own_content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="calculate", args={}))])
    contents = _to_contents(_messages_with_tool_round(GeminiTurn("model-a", own_content)), "model-a")

    assert [c.role for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[1].parts[0].text == "earlier answer"
    assert contents[3] is own_content  # replayed unchanged (keeps thought signatures)
    results = contents[4].parts
    assert [p.function_response.name for p in results] == ["calculate", "weekday_of"]
    assert results[0].function_response.id == "a"


def test_other_model_gets_calls_rebuilt_with_placeholder_signature():
    own_content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="calculate", args={}))])
    contents = _to_contents(_messages_with_tool_round(GeminiTurn("model-a", own_content)), "model-b")

    rebuilt = contents[3]
    assert rebuilt is not own_content
    assert [p.function_call.name for p in rebuilt.parts] == ["calculate", "weekday_of"]
    assert all(p.thought_signature == PLACEHOLDER_SIGNATURE for p in rebuilt.parts)


def test_all_tools_are_declared():
    declared = [d.name for d in _to_gemini_tools(TOOL_SPECS).function_declarations]
    assert declared == [t["name"] for t in TOOL_SPECS]


# ---------- cooldowns ----------

def _api_error(code: int, retry_delay: str | None = None, quota_id: str | None = None) -> errors.APIError:
    details = []
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    if quota_id:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]})
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X", "details": details}})


def test_cooldown_lengths():
    assert _cooldown(_api_error(429, "13s")) == 13
    assert _cooldown(_api_error(429, quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")) == gemini.DAILY_QUOTA_COOLDOWN_S
    assert _cooldown(_api_error(503)) == gemini.DEFAULT_COOLDOWN_S


# ---------- model cycling ----------

class _ScriptedModels:
    """Fake client.models: each model name maps to an error to raise, or None for success."""

    def __init__(self, outcomes: dict[str, Exception | None]):
        self.outcomes = outcomes
        self.calls: list[str] = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if error := self.outcomes.get(model):
            raise error
        part = types.Part.from_text(text=f'{{"answered_by": "{model}"}}')
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])


def _provider(models, outcomes, monkeypatch, clock=None):
    sleeps = []
    monkeypatch.setattr(gemini.time, "sleep", sleeps.append)
    if clock is not None:
        monkeypatch.setattr(gemini.time, "monotonic", lambda: clock[0])
    provider = GeminiProvider("fake-key", models)
    provider.client = type("Client", (), {"models": _ScriptedModels(outcomes)})()
    return provider, sleeps


def test_busy_model_switches_to_next_without_waiting(monkeypatch):
    provider, sleeps = _provider(["a", "b"], {"a": _api_error(429, "40s")}, monkeypatch)
    result = provider.generate("system", QUESTION, schema=dict)
    assert result.model == "b" and provider.model == "b"
    assert provider.client.models.calls == ["a", "b"]
    assert sleeps == []


def test_cooling_model_is_skipped_on_next_call(monkeypatch):
    clock = [1000.0]
    provider, _ = _provider(["a", "b"], {"a": _api_error(429, quota_id="PerDay")}, monkeypatch, clock)
    provider.generate("system", QUESTION, schema=dict)
    provider.client.models.calls.clear()

    provider.generate("system", QUESTION, schema=dict)
    assert provider.client.models.calls == ["b"]  # "a" is resting


def test_model_is_retried_after_cooldown(monkeypatch):
    clock = [1000.0]
    provider, _ = _provider(["a", "b"], {"a": _api_error(503)}, monkeypatch, clock)
    provider.generate("system", QUESTION, schema=dict)
    provider.client.models.outcomes = {}
    clock[0] += gemini.DEFAULT_COOLDOWN_S + 1

    assert provider.generate("system", QUESTION, schema=dict).model == "a"


def test_retired_model_is_dropped(monkeypatch):
    provider, _ = _provider(["old", "new"], {"old": _api_error(404)}, monkeypatch)
    assert provider.generate("system", QUESTION, schema=dict).model == "new"
    assert provider.cooldown_until["old"] == gemini.RETIRED


def test_all_busy_waits_briefly_once_then_succeeds(monkeypatch):
    clock = [1000.0]
    provider, sleeps = _provider(["a", "b"], {"a": _api_error(429, "5s"), "b": _api_error(429, "8s")}, monkeypatch, clock)

    def sleep_and_recover(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        provider.client.models.outcomes = {}

    monkeypatch.setattr(gemini.time, "sleep", sleep_and_recover)
    assert provider.generate("system", QUESTION, schema=dict).model == "a"
    assert sleeps == [5]


def test_all_busy_for_long_raises_busy(monkeypatch):
    daily = _api_error(429, quota_id="PerDay")
    provider, sleeps = _provider(["a", "b"], {"a": daily, "b": daily}, monkeypatch)
    with pytest.raises(ProviderBusyError, match="a: 429; b: 429"):
        provider.generate("system", QUESTION, schema=dict)
    assert sleeps == []  # an hour is too long to wait in a request


def test_other_errors_fail_immediately(monkeypatch):
    provider, _ = _provider(["a", "b"], {"a": _api_error(400)}, monkeypatch)
    with pytest.raises(ProviderError) as info:
        provider.generate("system", QUESTION, schema=dict)
    assert not isinstance(info.value, ProviderBusyError)
    assert provider.client.models.calls == ["a"]
