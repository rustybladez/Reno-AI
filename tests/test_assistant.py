"""End-to-end tests of ask() with the scripted FakeProvider (offline)."""

import json
from datetime import date

from leave_assistant.assistant import MAX_TOOL_ROUNDS, RETRY_INSTRUCTION, ask
from leave_assistant.guardrails import FALLBACK_TEXT
from leave_assistant.providers.base import ProviderResult, ToolCall
from leave_assistant.providers.fake import FakeProvider
from leave_assistant.schema import Turn

TODAY = date(2026, 10, 26)


def final_json(type="answer", answer="", citations=(), calculations=()) -> str:
    return json.dumps({
        "type": type,
        "answer": answer,
        "citations": [{"section_id": s, "quote": q} for s, q in citations],
        "uncovered_part": "",
        "calculations": list(calculations),
        "language": "en",
    })


def run(provider, question, policy, history=None):
    return ask(question, history=history, today=TODAY, provider=provider, policy=policy)


def test_simple_answer(example_policy):
    q = "How much holiday leave?"
    provider = FakeProvider({q: [final_json(answer="You get 99 days.", citations=[("2.1(a)", "99 days")])]})
    response = run(provider, q, example_policy)
    assert response.type == "answer"
    assert response.text.startswith("You get 99 days.")
    assert response.debug["unverified_numbers"] == []


def test_invalid_json_twice_becomes_fallback(example_policy):
    q = "Broken?"
    provider = FakeProvider({q: ["not json", "still not json"]})
    response = run(provider, q, example_policy)
    assert response.text == FALLBACK_TEXT
    assert response.debug["retried_invalid_json"] is True
    assert len(provider.calls) == 2


def test_retry_tells_the_model_what_went_wrong_and_debug_keeps_the_bad_replies(example_policy):
    q = "Broken twice?"
    provider = FakeProvider({q: ["Sure! The answer is 99 days.", ""]})
    response = run(provider, q, example_policy)

    retry_messages = provider.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "Sure! The answer is 99 days."}
    assert retry_messages[-1] == {"role": "user", "content": RETRY_INSTRUCTION}
    assert response.debug["invalid_outputs"] == ["Sure! The answer is 99 days.", "<empty>"]
    assert response.text == FALLBACK_TEXT


def test_invalid_json_then_valid_is_recovered(example_policy):
    q = "Retry?"
    provider = FakeProvider({q: ["oops", final_json(answer="You get 99 days.", citations=[("2.1(a)", "99 days")])]})
    assert run(provider, q, example_policy).type == "answer"
    assert provider.calls[1]["tools"] is None  # the retry asks for JSON only


def test_tool_call_then_final_answer(example_policy):
    q = "My basic is 45,000 and I have 12 unused days. How much?"
    provider = FakeProvider({q: [
        ProviderResult(tool_calls=[ToolCall("calculate", {"expression": "45000 / 30 * 12"}, id="1")]),
        final_json(
            answer="You would be paid 18,000.",
            citations=[("2.1(a)", "99 days")],
            calculations=["45,000 ÷ 30 × 12 = 18,000"],
        ),
    ]})
    response = run(provider, q, example_policy)

    assert response.debug["tool_calls"] == [
        {"name": "calculate", "args": {"expression": "45000 / 30 * 12"},
         "result": {"expression": "45000 / 30 * 12", "result": 18000,
                    "steps": ["45000 ÷ 30 = 1500", "1500 × 12 = 18000"]}},
    ]
    assert response.calculations == ["45,000 ÷ 30 × 12 = 18,000"]
    assert response.debug["unverified_numbers"] == []  # 18,000 comes from the tool
    # The second call saw the tool result.
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"]["result"] == 18000


def test_tool_loop_stops_at_max_rounds(example_policy):
    q = "Loop forever?"
    endless = ProviderResult(tool_calls=[ToolCall("weekday_of", {"date": "2026-10-26"})])
    provider = FakeProvider({q: [endless] * MAX_TOOL_ROUNDS + ["not json"]})
    response = run(provider, q, example_policy)

    with_tools = [c for c in provider.calls if c["tools"]]
    assert len(with_tools) == MAX_TOOL_ROUNDS
    assert provider.calls[MAX_TOOL_ROUNDS]["tools"] is None  # forced final answer
    assert len(response.debug["tool_calls"]) == MAX_TOOL_ROUNDS
    assert response.text == FALLBACK_TEXT


def test_tool_errors_are_passed_back_not_raised(example_policy):
    q = "Bad date?"
    provider = FakeProvider({q: [
        ProviderResult(tool_calls=[ToolCall("add_calendar_days", {"date": "2026-02-30", "days": 1})]),
        final_json(type="not_covered"),
    ]})
    response = run(provider, q, example_policy)
    assert "error" in response.debug["tool_calls"][0]["result"]
    assert response.text == FALLBACK_TEXT


def test_history_is_sent_and_numbers_from_history_pass(example_policy):
    q = "And for 3 days?"
    history = [Turn(role="user", content="My salary is 30,000"), Turn(role="assistant", content="Noted.")]
    provider = FakeProvider({q: [final_json(answer="With a salary of 30,000 for 3 days...", citations=[("2.1(a)", "99 days")])]})
    response = run(provider, q, example_policy, history=history)
    assert [m["content"] for m in provider.calls[0]["messages"]] == ["My salary is 30,000", "Noted.", q]
    assert response.debug["unverified_numbers"] == []


def test_numbers_from_earlier_assistant_answers_are_not_sources(example_policy):
    q = "And how much is that?"
    history = [Turn(role="user", content="How much leave?"), Turn(role="assistant", content="You get 45 days.")]
    provider = FakeProvider({q: [final_json(answer="That is 45 days.", citations=[("2.1(a)", "99 days")])]})
    response = run(provider, q, example_policy, history=history)
    assert response.debug["unverified_numbers"] == ["45"]  # the bot's own earlier number doesn't count


def test_debug_details(example_policy):
    q = "Hi"
    provider = FakeProvider({q: [final_json(type="smalltalk")]})
    debug = run(provider, q, example_policy).debug
    assert debug["provider"] == "fake / fake"
    assert debug["model_type"] == "smalltalk"
    assert debug["final_type"] == "smalltalk"
    assert "latency_s" in debug
