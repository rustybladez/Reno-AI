"""Offline tests of the eval runner's checks (no API calls)."""

import json
from datetime import datetime

from evals.run_evals import (
    PROJECT_ROOT,
    Case,
    CaseResult,
    check_case,
    contains,
    is_related_section,
    load_cases,
    run_case,
    score,
    targets_met,
    write_report,
)
from leave_assistant.guardrails import FALLBACK_TEXT, PRIVACY_REFUSAL_TEXT
from leave_assistant.providers.fake import FakeProvider
from leave_assistant.schema import AssistantResponse, Citation

EXAMPLE_CASES = PROJECT_ROOT / "evals" / "cases.example.yaml"


def response(type="answer", text="You get 99 days.", sections=("2.1(a)",), tools=(), unverified=()):
    return AssistantResponse(
        type=type,
        text=text,
        citations=[Citation(section_id=s, quote="q") for s in sections],
        debug={"tool_calls": [{"name": t, "args": {}, "result": {}} for t in tools],
               "unverified_numbers": list(unverified)},
    )


def case(**fields) -> Case:
    defaults = dict(id="T01", category="in_policy", question="q?", expected_type="answer")
    return Case(**(defaults | fields))


# ---------- matching ----------

def test_numbers_ignore_thousands_separators():
    assert contains("You will be paid 18000 BDT.", "18,000")
    assert contains("You will be paid 18,000.00 BDT.", "18000")


def test_numbers_match_whole_numbers_only():
    assert not contains("Effective 1 January 2026.", "20")
    assert not contains("You get 120 days.", "20")
    assert not contains("You earn 10.02 days.", "10")
    assert contains("You get 20 working days.", "20")


def test_dates_accept_several_formats():
    for text in ("by 19 November 2026", "by 19 Nov 2026", "by 2026-11-19", "by November 19, 2026",
                 "by Thursday, 19th November 2026"):
        assert contains(text, "19 November 2026"), text
    assert not contains("by 9 November 2026", "19 November 2026")


def test_text_match_is_case_insensitive():
    assert contains("Ask your department head.", "Department Head")


def test_related_sections():
    assert is_related_section("4.1(d)", "4.1(d)")
    assert is_related_section("4.1", "4.1(d)")
    assert is_related_section("4.1(d)", "4.1")
    assert not is_related_section("4.1(d)", "4.1(a)")
    assert not is_related_section("4.1", "4.10")


# ---------- check_case ----------

def test_passing_answer():
    result = check_case(case(expected_sections=["2.1"], must_include=["99"]), response())
    assert result.passed, result.reasons


def test_wrong_section_fails():
    result = check_case(case(expected_sections=["3"]), response())
    assert not result.passed and "cited" in result.reasons[0]


def test_not_covered_must_be_exact_fallback():
    assert check_case(case(expected_type="not_covered"), response("not_covered", FALLBACK_TEXT, ())).passed
    result = check_case(case(expected_type="not_covered"), response("not_covered", FALLBACK_TEXT + " ", ()))
    assert not result.passed


def test_privacy_must_be_exact_template():
    good = response("privacy_refusal", PRIVACY_REFUSAL_TEXT + "\n\n**Source:** Section 9.1 – X", ())
    assert check_case(case(expected_type="privacy_refusal"), good).passed
    altered = response("privacy_refusal", "Sorry, I can't share that.", ())
    assert not check_case(case(expected_type="privacy_refusal"), altered).passed
    assert not check_case(case(expected_type="privacy_refusal"), response("answer", "No.", ())).passed


def test_partial_needs_fallback_text():
    ok = response("partial", "You get 8 days.\n\nFor the rest of your question: " + FALLBACK_TEXT)
    assert check_case(case(expected_type="partial"), ok).passed
    assert not check_case(case(expected_type="partial"), response("partial", "You get 8 days.")).passed


def test_must_not_include_and_any_of():
    assert not check_case(case(must_not_include=["99"]), response()).passed
    assert check_case(case(must_include=[["৯৯", "99"]]), response()).passed


def test_required_tool_missing_fails():
    assert not check_case(case(must_call_tools=["calculate"]), response()).passed
    assert check_case(case(must_call_tools=["calculate"]), response(tools=["calculate"])).passed


def test_unverified_number_is_a_failure():
    result = check_case(case(), response(unverified=["42"]))
    assert not result.passed and result.provenance_failed


def test_soft_check_is_only_a_warning():
    soft = [{"if_includes": "99 days", "require_any": ["assum"]}]
    result = check_case(case(soft_checks=soft), response())
    assert result.passed and result.warnings


# ---------- targets ----------

def result(category, passed, optional=False):
    return CaseResult(case(category=category, optional=optional), passed, [] if passed else ["x"], [])


def test_targets():
    results = [result("in_policy", True)] * 19 + [result("not_covered", True)]
    assert targets_met(results)[0]
    # A single privacy failure misses the 100% target even when overall is >= 95%.
    met, misses = targets_met(results + [result("privacy", False)] + [result("in_policy", True)] * 10)
    assert not met and any("privacy" in m for m in misses)


def test_optional_cases_are_not_counted():
    assert targets_met([result("in_policy", True), result("bangla", False, optional=True)])[0]


def test_holdout_is_scored_separately_and_not_targeted():
    holdout_fail = CaseResult(case(id="K01", category="privacy", holdout=True), False, ["x"], [])
    results = [result("privacy", True), holdout_fail]

    _, main = score(results)
    _, held = score(results, holdout=True)
    assert (main.passed, main.total) == (1, 1)
    assert (held.passed, held.total) == (0, 1)
    assert targets_met(results)[0]  # a hold-out miss is reported, not a target miss


def test_report_records_settings_and_holdout(tmp_path):
    holdout_pass = CaseResult(case(id="K01", holdout=True), True, [], [], response=response())
    report = tmp_path / "report.md"
    write_report([holdout_pass], "fake / m", "gemini: models a → b; thinking level medium", 3,
                 datetime(2026, 10, 26, 12, 0), report)
    text = report.read_text(encoding="utf-8")
    assert "thinking level medium" in text
    assert "Hold-out (never used for prompt tuning): 1/1" in text
    assert "partial run" in text  # 1 of 3 cases


# ---------- end to end with the example cases ----------

def final_json(type, answer="", citations=()):
    return json.dumps({"type": type, "answer": answer, "uncovered_part": "", "calculations": [],
                       "citations": [{"section_id": s, "quote": q} for s, q in citations], "language": "en"})


def test_example_cases_run_offline(example_policy):
    cases = {c.id: c for c in load_cases(EXAMPLE_CASES)}
    provider = FakeProvider({
        cases["X01"].question: [final_json("answer", "You get 99 days.", [("2.1(a)", "99 days of Holiday Leave")])],
        cases["X02"].question: [final_json("not_covered")],
        cases["X03"].question: [final_json("privacy_refusal")],
    })
    for c in cases.values():
        assert run_case(c, provider, example_policy).passed, c.id


def test_real_cases_file_is_valid_if_present():
    path = PROJECT_ROOT / "evals" / "cases.yaml"
    if path.exists():  # gitignored, so absent in a fresh clone
        cases = load_cases(path)
        assert all(c.expected_type in {"answer", "partial", "not_covered", "privacy_refusal", "smalltalk"}
                   for c in cases)
