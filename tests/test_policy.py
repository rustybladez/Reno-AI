import json
from dataclasses import replace
from datetime import date

import pytest

from leave_assistant.config import load_settings
from leave_assistant.policy import POLICY_NOT_FOUND_MESSAGE, PolicyNotFoundError, load_policy
from leave_assistant.prompt import build_messages, build_system_prompt, format_today
from leave_assistant.schema import Turn

from .conftest import EXAMPLE_POLICY_PATH


def _settings(**changes):
    return replace(load_settings(), **changes)


# ---------- loader ----------

def test_loads_from_secret_string_first(tmp_path):
    data = json.loads(EXAMPLE_POLICY_PATH.read_text(encoding="utf-8"))
    data["meta"]["title"] = "From the secret"
    settings = _settings(policy_json=json.dumps(data), policy_path=EXAMPLE_POLICY_PATH)
    assert load_policy(settings).meta["title"] == "From the secret"


def test_loads_from_file():
    policy = load_policy(_settings(policy_json=None, policy_path=EXAMPLE_POLICY_PATH))
    assert policy.get("2.1(a)").heading == "Holiday Leave"


def test_missing_policy_gives_clear_error(tmp_path):
    settings = _settings(policy_json=None, policy_path=tmp_path / "missing.json")
    with pytest.raises(PolicyNotFoundError, match="Policy file not found"):
        load_policy(settings)
    assert POLICY_NOT_FOUND_MESSAGE == "Policy file not found. See README → Setup."


# ---------- lookups ----------

def test_labels(example_policy):
    assert example_policy.label("2.1(a)") == "Section 2.1(a) – Holiday Leave"
    assert example_policy.label("2.1") == "Section 2.1 – Holiday Leave"  # group id from the path
    assert example_policy.label("2") == "Section 2 – Types of Leave"
    assert example_policy.label("header") == "Policy details (EX-POL-001)"


def test_descendants(example_policy):
    assert [s.id for s in example_policy.descendants("2.1")] == ["2.1(a)"]
    assert [s.id for s in example_policy.descendants("2")] == ["2.1(a)"]
    assert example_policy.descendants("9") == []


def test_render_for_prompt(example_policy):
    rendered = example_policy.render_for_prompt()
    assert "[2.1(a)] (2.1 Holiday Leave) Employees receive 99 days" in rendered


# ---------- prompt ----------

def test_system_prompt_has_policy_and_today(example_policy):
    prompt = build_system_prompt(example_policy, date(2026, 10, 26))
    assert "Today's date is Monday 26 October 2026" in prompt
    assert "[2.1(a)] (2.1 Holiday Leave)" in prompt
    assert "{policy}" not in prompt and "{today}" not in prompt


def test_format_today():
    assert format_today(date(2026, 11, 1)) == "Sunday 1 November 2026"


def test_build_messages_keeps_last_six_turns():
    history = [Turn(role="user" if i % 2 == 0 else "assistant", content=f"m{i}") for i in range(10)]
    messages = build_messages("new question", history)
    assert [m["content"] for m in messages] == ["m4", "m5", "m6", "m7", "m8", "m9", "new question"]
