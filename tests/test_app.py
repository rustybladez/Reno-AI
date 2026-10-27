"""Offline UI tests with Streamlit's AppTest, the FakeProvider and the dummy policy."""

import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import leave_assistant.providers as providers
from leave_assistant.assistant import get_policy
from leave_assistant.config import PROJECT_ROOT
from leave_assistant.guardrails import FALLBACK_TEXT
from leave_assistant.providers.base import ProviderBusyError
from leave_assistant.providers.fake import FakeProvider

from .conftest import EXAMPLE_POLICY_PATH

APP = str(PROJECT_ROOT / "app.py")


def final_json(type, answer="", citations=()):
    return json.dumps({"type": type, "answer": answer, "uncovered_part": "", "calculations": [],
                       "citations": [{"section_id": s, "quote": q} for s, q in citations], "language": "en"})


@pytest.fixture
def app_with(monkeypatch):
    """Returns a function that starts the app with a given provider and the dummy policy."""
    monkeypatch.setenv("POLICY_PATH", str(EXAMPLE_POLICY_PATH))
    monkeypatch.setenv("POLICY_JSON", "")
    # Fake keys, so the tests never depend on a developer's .env (a fresh clone has none).
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "")

    def start(provider):
        get_policy.cache_clear()
        st.cache_resource.clear()
        monkeypatch.setattr(providers, "get_provider", lambda settings: provider)
        return AppTest.from_file(APP, default_timeout=30).run()

    yield start
    get_policy.cache_clear()
    st.cache_resource.clear()


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_page_renders_with_sidebar(app_with):
    at = app_with(FakeProvider({}))
    assert not at.exception
    assert at.title[0].value == "Reno AI — Leave Policy Assistant"
    assert len(at.sidebar.button) >= 9  # sample questions + "Clear chat"
    assert any("Today:" in m.value for m in at.sidebar.markdown)
    assert any("Conversations are not stored." in c.value for c in at.caption)


def test_typed_question_gets_cited_answer_and_policy_text(app_with):
    q = "How much holiday leave?"
    at = app_with(FakeProvider({q: [final_json("answer", "You get 99 days.", [("2.1(a)", "99 days")])]}))
    at.chat_input[0].set_value(q).run()

    assert not at.exception
    text = all_markdown(at)
    assert "You get 99 days." in text
    assert "**Source:** Section 2.1(a) – Holiday Leave" in text
    assert "Employees receive 99 days of Holiday Leave per year." in text  # "Policy text" expander
    assert [e.label for e in at.expander] == ["Policy text"]


def test_sample_button_sends_its_question(app_with):
    q = "Is there study or exam leave?"
    at = app_with(FakeProvider({q: [final_json("not_covered")]}))
    next(b for b in at.sidebar.button if b.label == q).click().run()

    answer, contact = [m.value for m in at.chat_message[1].markdown]
    assert answer == FALLBACK_TEXT  # the fallback sentence stays exact
    assert contact == "HR email: [people@example.invalid](mailto:people@example.invalid)"


def test_hr_email_only_under_fallback_answers(app_with):
    q = "How much holiday leave?"
    at = app_with(FakeProvider({q: [final_json("answer", "You get 99 days.", [("2.1(a)", "99 days")])]}))
    at.chat_input[0].set_value(q).run()
    assert "mailto:" not in all_markdown(at)


def test_welcome_starters_show_on_empty_chat_and_send_their_question(app_with):
    q = "How many leave days has Rahim taken?"
    at = app_with(FakeProvider({q: [final_json("privacy_refusal")]}))
    starters = [b for b in at.main.button if b.key and b.key.startswith("welcome_")]
    assert len(starters) == 3

    next(b for b in starters if b.label == q).click().run()
    assert not at.exception
    assert [m.value for m in at.chat_message[0].markdown] == [q]
    assert not [b for b in at.main.button if b.key and b.key.startswith("welcome_")]  # gone once chatting


def test_question_length_is_limited(app_with):
    at = app_with(FakeProvider({}))
    assert at.chat_input[0].max_chars == 500


def test_debug_toggle_shows_debug_details(app_with):
    q = "Hi"
    at = app_with(FakeProvider({q: [final_json("smalltalk")]}))
    at.sidebar.toggle[0].set_value(True).run()
    at.chat_input[0].set_value(q).run()
    assert "Debug details" in [e.label for e in at.expander]


def test_busy_provider_shows_friendly_message(app_with):
    class BusyProvider:
        name, model = "busy", "busy"

        def generate(self, *args, **kwargs):
            raise ProviderBusyError("all models busy")

    at = app_with(BusyProvider())
    at.chat_input[0].set_value("Anything?").run()
    assert not at.exception
    assert "busy right now" in at.warning[0].value


def test_without_any_key_the_app_asks_for_one_and_uses_it_for_the_session(app_with, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")  # e.g. a local run with no .env file
    built_with = []
    q = "How much holiday leave?"
    provider = FakeProvider({q: [final_json("answer", "You get 99 days.", [("2.1(a)", "99 days")])]})

    at = app_with(provider)
    monkeypatch.setattr(providers, "get_provider", lambda settings: built_with.append(settings) or provider)
    assert "An API key is needed" in at.info[0].title
    assert not at.chat_input  # no chat until a key is given

    at.text_input[0].set_value("  pasted-gemini-key  ")
    at.button[0].click().run()
    assert built_with[-1].gemini_api_key == "pasted-gemini-key"  # trimmed, used for this session

    at.chat_input[0].set_value(q).run()
    assert not at.exception
    assert "You get 99 days." in all_markdown(at)
    assert any(b.label == "Use a different API key" for b in at.sidebar.button)


def test_key_form_needs_at_least_one_key(app_with, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    at = app_with(FakeProvider({}))
    at.button[0].click().run()
    assert "at least one key" in at.warning[0].value
    assert not at.chat_input


def test_missing_policy_shows_setup_message(app_with, monkeypatch, tmp_path):
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "missing.json"))
    at = app_with(FakeProvider({}))
    assert "Policy file not found" in at.error[0].value
