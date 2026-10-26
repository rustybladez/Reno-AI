from leave_assistant.guardrails import (
    FALLBACK_TEXT,
    PRIVACY_REFUSAL_TEXT,
    SMALLTALK_TEXT,
    apply_guardrails,
    parse_output,
    quote_matches,
    unverified_numbers,
)
from leave_assistant.policy import parse_policy
from leave_assistant.schema import Citation, LLMOutput

GOOD_QUOTE = "Employees receive 99 days of Holiday Leave per year."


def output(type="answer", answer="You get 99 days.", citations=None, **extra) -> LLMOutput:
    if citations is None:
        citations = [Citation(section_id="2.1(a)", quote=GOOD_QUOTE)]
    fields = dict(uncovered_part="", calculations=[], language="en")
    fields.update(extra)
    return LLMOutput(type=type, answer=answer, citations=citations, **fields)


# ---------- fixed texts ----------

def test_fallback_is_byte_exact(example_policy):
    response = apply_guardrails(output(type="not_covered", answer="", citations=[]), example_policy)
    assert response.text == "I don't know, please contact HR."
    assert response.text == FALLBACK_TEXT
    assert response.type == "not_covered"


def test_fallback_ignores_any_model_text(example_policy):
    response = apply_guardrails(output(type="not_covered", answer="I think maybe 5 days?"), example_policy)
    assert response.text == FALLBACK_TEXT


def test_privacy_rendering_equals_template(example_policy):
    # The example policy has none of the confidentiality sections, so no Source line.
    # The HR email comes from the policy's meta.hr_contact, not from the code.
    response = apply_guardrails(output(type="privacy_refusal", answer="", citations=[]), example_policy)
    assert response.text == PRIVACY_REFUSAL_TEXT + " at people@example.invalid."
    assert response.type == "privacy_refusal"


def test_privacy_refusal_cites_confidentiality_sections_that_exist():
    policy = parse_policy({"meta": {}, "sections": [
        {"id": "9.1", "parent": "9", "heading": "Confidentiality of Leave Records",
         "path": "9. Confidentiality of Leave Records", "text": "Leave records are confidential."},
        {"id": "11.2", "parent": "11", "heading": "Policy Review and Contact",
         "path": "11. Policy Review and Contact", "text": "Contact HR."},
    ]})
    response = apply_guardrails(output(type="privacy_refusal", answer="", citations=[]), policy)
    assert response.text == (  # no hr_contact in this policy's meta: the sentence just ends
        PRIVACY_REFUSAL_TEXT + ".\n\n**Source:** Section 9.1 – Confidentiality of Leave Records; "
        "Section 11.2 – Policy Review and Contact"
    )
    assert [c.section_id for c in response.citations] == ["9.1", "11.2"]
    assert response.citations[0].quote == "Leave records are confidential."


def test_smalltalk_rendering(example_policy):
    response = apply_guardrails(output(type="smalltalk", answer="", citations=[]), example_policy)
    assert response.text == SMALLTALK_TEXT


def test_invalid_output_becomes_fallback(example_policy):
    assert apply_guardrails(None, example_policy).text == FALLBACK_TEXT


# ---------- citations ----------

def test_valid_answer_is_rendered_with_source(example_policy):
    response = apply_guardrails(output(), example_policy)
    assert response.type == "answer"
    assert response.text == "You get 99 days.\n\n**Source:** Section 2.1(a) – Holiday Leave"


def test_invalid_section_id_is_dropped(example_policy):
    citations = [
        Citation(section_id="7.7", quote=GOOD_QUOTE),
        Citation(section_id="2.1(a)", quote=GOOD_QUOTE),
    ]
    response = apply_guardrails(output(citations=citations), example_policy)
    assert response.type == "answer"
    assert [c.section_id for c in response.citations] == ["2.1(a)"]
    assert "7.7" not in response.text


def test_all_invalid_section_ids_become_fallback(example_policy):
    response = apply_guardrails(output(citations=[Citation(section_id="7.7", quote=GOOD_QUOTE)]), example_policy)
    assert response.text == FALLBACK_TEXT


def test_fabricated_quote_becomes_fallback(example_policy):
    fabricated = [Citation(section_id="2.1(a)", quote="Employees may work from home on Fridays.")]
    response = apply_guardrails(output(citations=fabricated), example_policy)
    assert response.text == FALLBACK_TEXT


def test_quote_with_changed_number_is_rejected_even_if_close():
    assert quote_matches("Employees receive 98 days of Holiday Leave per year", GOOD_QUOTE) is None


def test_parent_id_citation_with_quote_in_child_is_valid(example_policy):
    response = apply_guardrails(output(citations=[Citation(section_id="2.1", quote="99 days of Holiday Leave")]), example_policy)
    assert response.type == "answer"
    assert response.citations[0].section_id == "2.1(a)"


def test_section_id_variants_are_accepted(example_policy):
    for section_id in ("Section 2.1 (a)", "[2.1(a)]", "§2.1(a)"):
        response = apply_guardrails(output(citations=[Citation(section_id=section_id, quote=GOOD_QUOTE)]), example_policy)
        assert response.type == "answer", section_id


def test_quote_matching_tolerates_case_quotes_dashes_and_ellipsis():
    assert quote_matches("EMPLOYEES RECEIVE 99 DAYS", GOOD_QUOTE) == "exact"
    assert quote_matches("Employees receive ... per year", GOOD_QUOTE) == "exact"
    assert quote_matches("Carried–forward days", "Carried-forward days must be used") == "exact"
    assert quote_matches("Employees recieve 99 days of Holiday Leave", GOOD_QUOTE) == "fuzzy"


def test_quote_matching_in_table():
    table = "| Leave type | Entitlement |\n|---|---|\n| Holiday Leave | 99 days per year |"
    assert quote_matches("Holiday Leave | 99 days per year", table) == "exact"
    assert quote_matches("Holiday Leave 99 days per year", table) == "exact"


def test_empty_answer_becomes_fallback(example_policy):
    assert apply_guardrails(output(answer="  "), example_policy).text == FALLBACK_TEXT


def test_sources_are_deduplicated(example_policy):
    citations = [
        Citation(section_id="2.1(a)", quote="99 days"),
        Citation(section_id="2.1(a)", quote="Holiday Leave per year"),
        Citation(section_id="3", quote="Leave records are confidential"),
    ]
    response = apply_guardrails(output(citations=citations), example_policy)
    assert response.text.endswith(
        "**Source:** Section 2.1(a) – Holiday Leave; Section 3 – Confidentiality"
    )


# ---------- partial ----------

def test_partial_contains_fallback_verbatim(example_policy):
    response = apply_guardrails(output(type="partial", uncovered_part="half days"), example_policy)
    assert response.type == "partial"
    assert FALLBACK_TEXT in response.text
    assert response.text.endswith("For the rest of your question: I don't know, please contact HR.")


# ---------- JSON parsing ----------

def test_parse_output_accepts_fenced_json_and_rejects_garbage():
    raw = '{"type": "smalltalk", "answer": "", "citations": [], "uncovered_part": "", "calculations": [], "language": "en"}'
    assert parse_output(raw).type == "smalltalk"
    assert parse_output(f"```json\n{raw}\n```").type == "smalltalk"
    assert parse_output("not json") is None
    assert parse_output(f"Here is my answer: {raw} Hope that helps!").type == "smalltalk"  # prose around it
    assert parse_output('Here: {"type": "answer"} done') is None  # extracted object must still match the schema
    assert parse_output('{"type": "answer"}') is None
    assert parse_output(None) is None


# ---------- number provenance ----------

def test_number_from_tool_result_passes():
    sources = ['{"name": "calculate", "result": {"result": 18000}}']
    assert unverified_numbers("You will be paid 18,000 BDT.", sources) == []


def test_invented_number_is_flagged():
    assert unverified_numbers("You will be paid 25,000 BDT.", ["basic salary divided by 30"]) == ["25,000"]


def test_rounded_tool_result_passes():
    sources = ['{"result": 8333.333333}']
    assert unverified_numbers("That is 8,333.33 BDT.", sources) == []


def test_section_references_are_exempt():
    answer = "See Section 4.1, 4.1(d), §8.1 and Sections 4.3 and 8.1 for details."
    assert unverified_numbers(answer, []) == []


def test_bare_number_matching_a_section_id_is_not_exempt():
    # "4.1 days" is not a section reference, even though 4.1 is a section id.
    assert unverified_numbers("You get 4.1 days.", []) == ["4.1"]
    assert unverified_numbers("Under Section 4.1 and 20 days later", []) == ["20"]


def test_dates_from_tool_results_pass():
    sources = ['{"date": "2026-11-19", "display": "19 November 2026"}']
    assert unverified_numbers("The last day is 19 November 2026.", sources) == []


def test_unverified_numbers_reported_in_debug_only(example_policy):
    response = apply_guardrails(output(answer="You get 99 days, or 120 if you ask nicely."), example_policy)
    assert response.type == "answer"  # not blocked in the UI
    assert response.debug["unverified_numbers"] == ["120"]
