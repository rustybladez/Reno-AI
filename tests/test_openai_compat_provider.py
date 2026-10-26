"""Offline tests for the OpenAI-compatible (DeepSeek) provider and the provider chain."""

import json
from dataclasses import replace

import httpx
import pytest

from leave_assistant.config import load_settings
from leave_assistant.providers import get_provider
from leave_assistant.providers.base import (
    ProviderBusyError,
    ProviderConfigError,
    ProviderError,
    ProviderResult,
    ToolCall,
)
from leave_assistant.providers.chain import ChainProvider
from leave_assistant.providers.gemini import GeminiTurn
from leave_assistant.providers.openai_compat import OpenAICompatProvider, OpenAICompatTurn, _to_openai_messages
from leave_assistant.schema import LLMOutput
from leave_assistant.tools import TOOL_SPECS, dispatch

QUESTION = [{"role": "user", "content": "q"}]


def provider_with(handler, **kwargs) -> tuple[OpenAICompatProvider, list[dict]]:
    """Provider whose HTTP calls go to `handler`; returns it and the list of request bodies sent."""
    sent: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    provider = OpenAICompatProvider("https://api.example.com", "key", "deepseek-flash", **kwargs)
    provider.client = httpx.Client(transport=httpx.MockTransport(record))
    return provider, sent


def reply(message: dict, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})


def test_missing_settings_is_a_config_error():
    with pytest.raises(ProviderConfigError):
        OpenAICompatProvider("https://api.example.com", None, "deepseek-flash")


def test_request_has_json_mode_tools_schema_and_thinking():
    provider, sent = provider_with(lambda r: reply({"role": "assistant", "content": '{"type": "smalltalk"}'}),
                                   thinking="enabled", reasoning_effort="low")
    result = provider.generate("SYSTEM", QUESTION, LLMOutput, tools=TOOL_SPECS)

    body = sent[0]
    assert result.text == '{"type": "smalltalk"}' and result.model == "deepseek-flash"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "low"
    assert [t["function"]["name"] for t in body["tools"]] == [t["name"] for t in TOOL_SPECS]
    system = body["messages"][0]
    assert system["role"] == "system" and system["content"].startswith("SYSTEM")
    assert '"uncovered_part"' in system["content"]  # the schema is described in the prompt


def test_thinking_fields_are_omitted_when_not_configured():
    provider, sent = provider_with(lambda r: reply({"role": "assistant", "content": "{}"}))
    provider.generate("S", QUESTION, LLMOutput)
    assert "thinking" not in sent[0] and "reasoning_effort" not in sent[0] and "tools" not in sent[0]


def test_tool_calls_are_parsed_and_bad_arguments_become_tool_errors():
    message = {"role": "assistant", "content": "", "reasoning_content": "thinking...", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}},
        {"id": "c2", "type": "function", "function": {"name": "calculate", "arguments": "{not json"}},
    ]}
    provider, _ = provider_with(lambda r: reply(message, "tool_calls"))
    result = provider.generate("S", QUESTION, LLMOutput, tools=TOOL_SPECS)

    assert [(c.name, c.args, c.id) for c in result.tool_calls][0] == ("calculate", {"expression": "1+1"}, "c1")
    assert "error" in dispatch(result.tool_calls[1].name, result.tool_calls[1].args)
    assert isinstance(result.raw, OpenAICompatTurn) and result.raw.message["reasoning_content"] == "thinking..."


def test_own_turn_is_replayed_with_reasoning_content():
    own = {"role": "assistant", "content": "", "reasoning_content": "why", "tool_calls": [{"id": "c1"}]}
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [ToolCall("calculate", {}, id="c1")], "raw": OpenAICompatTurn("deepseek-flash", own)},
        {"role": "tool", "id": "c1", "name": "calculate", "content": {"result": 2}},
    ]
    converted = _to_openai_messages(messages, "deepseek-flash")
    assert converted[1] == own
    assert converted[2] == {"role": "tool", "tool_call_id": "c1", "content": '{"result": 2}'}


def test_calls_from_another_provider_are_rebuilt_with_matching_ids():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [ToolCall("calculate", {"expression": "1"}), ToolCall("weekday_of", {})],
         "raw": GeminiTurn("gemini-x", None)},
        {"role": "tool", "id": None, "name": "calculate", "content": {"result": 1}},
        {"role": "tool", "id": None, "name": "weekday_of", "content": {"weekday": "Monday"}},
    ]
    converted = _to_openai_messages(messages, "deepseek-flash")
    call_ids = [c["id"] for c in converted[1]["tool_calls"]]
    assert converted[1]["role"] == "assistant" and "reasoning_content" not in converted[1]
    assert [converted[2]["tool_call_id"], converted[3]["tool_call_id"]] == call_ids
    assert json.loads(converted[1]["tool_calls"][0]["function"]["arguments"]) == {"expression": "1"}


@pytest.mark.parametrize("status", [402, 429, 500, 503])
def test_busy_statuses_fail_fast(status):
    provider, sent = provider_with(lambda r: httpx.Response(status, text="busy"))
    with pytest.raises(ProviderBusyError):
        provider.generate("S", QUESTION, LLMOutput)
    assert len(sent) == 1  # no retries: the chain moves on immediately


def test_timeout_is_busy_and_bad_request_is_an_error():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider, _ = provider_with(timeout)
    with pytest.raises(ProviderBusyError):
        provider.generate("S", QUESTION, LLMOutput)

    provider, _ = provider_with(lambda r: httpx.Response(400, text="bad"))
    with pytest.raises(ProviderError) as info:
        provider.generate("S", QUESTION, LLMOutput)
    assert not isinstance(info.value, ProviderBusyError)


def test_insufficient_resources_finish_reason_is_busy():
    provider, _ = provider_with(lambda r: reply({"role": "assistant", "content": ""}, "insufficient_system_resource"))
    with pytest.raises(ProviderBusyError):
        provider.generate("S", QUESTION, LLMOutput)


def test_keep_alive_blank_lines_before_json_are_ignored():
    body = "\n\n\n" + json.dumps({"choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}]})
    provider, _ = provider_with(lambda r: httpx.Response(200, text=body))
    assert provider.generate("S", QUESTION, LLMOutput).text == "{}"


# ---------- chain ----------

class Scripted:
    def __init__(self, name, outcome):
        self.name, self.model, self.outcome, self.calls = name, f"{name}-model", outcome, 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_chain_falls_back_when_busy_and_reports_current_provider():
    first = Scripted("a", ProviderBusyError("429"))
    second = Scripted("b", ProviderResult(text="{}", model="b-model"))
    chain = ChainProvider([first, second])
    assert chain.generate("S", QUESTION, LLMOutput).text == "{}"
    assert (chain.name, chain.model) == ("b", "b-model")


def test_chain_raises_busy_when_all_busy_and_passes_other_errors_through():
    chain = ChainProvider([Scripted("a", ProviderBusyError("x")), Scripted("b", ProviderBusyError("y"))])
    with pytest.raises(ProviderBusyError, match="All providers busy"):
        chain.generate("S", QUESTION, LLMOutput)

    second = Scripted("b", ProviderResult(text="{}"))
    chain = ChainProvider([Scripted("a", ProviderError("bad request")), second])
    with pytest.raises(ProviderError):
        chain.generate("S", QUESTION, LLMOutput)
    assert second.calls == 0  # a real error is not hidden by falling back


# ---------- get_provider ----------

def _settings(**changes):
    base = dict(gemini_api_key="g", openai_compat_base_url="https://api.example.com",
                openai_compat_api_key="k", openai_compat_model="deepseek-flash")
    return replace(load_settings(), **(base | changes))


def test_get_provider_builds_primary_then_fallback():
    deepseek_first = get_provider(_settings(llm_provider="openai_compat"))
    assert [p.name for p in deepseek_first.providers] == ["openai_compat", "gemini"]

    gemini_first = get_provider(_settings(llm_provider="gemini"))
    assert [p.name for p in gemini_first.providers] == ["gemini", "openai_compat"]


def test_get_provider_uses_whichever_keys_are_set():
    only_gemini = get_provider(_settings(llm_provider="openai_compat", openai_compat_api_key=None))
    assert only_gemini.name == "gemini" and not isinstance(only_gemini, ChainProvider)

    only_deepseek = get_provider(_settings(llm_provider="openai_compat", gemini_api_key=None))
    assert only_deepseek.name == "openai_compat" and not isinstance(only_deepseek, ChainProvider)

    with pytest.raises(ProviderConfigError, match="No API key"):
        get_provider(_settings(openai_compat_api_key=None, gemini_api_key=None))


def test_get_provider_without_fallback_and_unknown_name():
    single = get_provider(_settings(llm_provider="openai_compat"), fallback=False)
    assert single.name == "openai_compat" and not isinstance(single, ChainProvider)

    with pytest.raises(ProviderConfigError):
        get_provider(_settings(llm_provider="nope"))
