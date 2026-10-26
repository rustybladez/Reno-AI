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
    q = "My cousin passed away. How many days of bereavement leave do I get?"
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


def test_missing_policy_shows_setup_message(app_with, monkeypatch, tmp_path):
    monkeypatch.setenv("POLICY_PATH", str(tmp_path / "missing.json"))
    at = app_with(FakeProvider({}))
    assert "Policy file not found" in at.error[0].value
