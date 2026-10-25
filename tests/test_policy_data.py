"""Sanity checks on the hand-verified data/policy.json.

The real policy is gitignored, so these tests are skipped when it isn't present.
"""

import json
import re

import pytest

from leave_assistant.config import PROJECT_ROOT

POLICY_PATH = PROJECT_ROOT / "data" / "policy.json"

pytestmark = pytest.mark.skipif(not POLICY_PATH.exists(), reason="data/policy.json not present")

# Real words in the policy that look like extraction artefacts.
ALLOWED_CAMEL_CASE = {"PeopleHub"}


@pytest.fixture(scope="module")
def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_meta_complete(policy):
    for field in ("title", "policy_number", "version", "effective_date", "owner", "applies_to", "hr_contact"):
        assert policy["meta"][field], f"meta.{field} is empty"


def test_ids_unique(policy):
    ids = [s["id"] for s in policy["sections"]]
    assert len(ids) == len(set(ids))


def test_no_empty_fields(policy):
    for s in policy["sections"]:
        for field in ("id", "heading", "path", "text"):
            assert s[field].strip(), f"{s['id']}: empty {field}"


def test_all_top_level_sections_present(policy):
    top_levels = {re.match(r"\d+", s["id"]).group() for s in policy["sections"] if s["id"] != "header"}
    assert top_levels == {str(n) for n in range(1, 12)}
    assert any(s["id"] == "header" for s in policy["sections"])


def test_parent_matches_id(policy):
    for s in policy["sections"]:
        if m := re.fullmatch(r"(\d+\.\d+)\([a-z]\)", s["id"]):
            assert s["parent"] == m.group(1), s["id"]
        elif m := re.fullmatch(r"(\d+)\.\d+", s["id"]):
            assert s["parent"] == m.group(1), s["id"]
        else:
            assert s["parent"] == "", s["id"]


def test_no_footer_or_end_marker_remnants(policy):
    for s in policy["sections"]:
        assert not re.search(r"Page \d+ of \d+", s["text"]), s["id"]
        assert "Leave Policy (Sample)" not in s["text"], s["id"]
        assert "End of policy" not in s["text"], s["id"]


def test_no_joined_word_artefacts(policy):
    for s in policy["sections"]:
        for field in ("heading", "path", "text"):
            text = s[field]
            for word in re.findall(r"\S*[a-z][A-Z]\S*", text):
                assert word.strip(".,()") in ALLOWED_CAMEL_CASE, f"{s['id']} {field}: {word!r}"
            assert not re.search(r"\d[A-Za-z]{2,}", text), f"{s['id']} {field}: digit joined to word"


def test_tables_in_the_right_sections(policy):
    by_id = {s["id"]: s for s in policy["sections"]}
    assert by_id["4.0"]["text"].startswith("| Leave type |")
    assert by_id["10"]["text"].startswith("| Role |")
    tables = [s["id"] for s in policy["sections"] if "|---" in s["text"]]
    assert tables == ["header", "4.0", "10"]
