"""Evaluation runner (PRD §12): runs every case through ask() and checks the result.

    uv run python -m evals.run_evals [--provider gemini] [--model X] [--no-fallback] [--only A01,B03] [--delay 13]

All checks are deterministic (no LLM-as-judge). Writes a console summary and a Markdown
report to evals/results/<provider>_<model>_<timestamp>.md. Exit code 0 = targets met.
"""

import argparse
import dataclasses
import hashlib
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import yaml

from leave_assistant.assistant import ask
from leave_assistant.config import PROJECT_ROOT, TIMEZONE, load_settings
from leave_assistant.guardrails import FALLBACK_TEXT, PRIVACY_REFUSAL_TEXT
from leave_assistant.policy import Policy, PolicyNotFoundError, load_policy
from leave_assistant.prompt import SYSTEM_PROMPT_PATH
from leave_assistant.providers import get_provider
from leave_assistant.providers.base import Provider, ProviderBusyError, ProviderError
from leave_assistant.schema import AssistantResponse

EVAL_TODAY = date(2026, 10, 26)  # Monday; every run uses the same date (§12.1)
DEFAULT_CASES = PROJECT_ROOT / "evals" / "cases.yaml"
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DEFAULT_DELAY_S = 13.0  # between cases; the free tier allows ~5 requests per minute per model
BUSY_RETRIES = 2  # extra attempts for a case when every model is busy
BUSY_WAIT_S = 60.0
MAX_CONSECUTIVE_BUSY = 3  # then stop: the daily quotas are probably used up

# §12.4. Categories not listed here have no target of their own (they count towards overall).
CATEGORY_TARGETS = {"not_covered": 1.0, "privacy": 1.0, "injection": 1.0, "calculation": 0.9}
OVERALL_TARGET = 0.95


# ---------- cases ----------

@dataclass
class Case:
    id: str
    category: str
    question: str
    expected_type: str
    expected_sections: list[str] = field(default_factory=list)
    must_include: list[str | list[str]] = field(default_factory=list)  # a nested list = any of
    must_not_include: list[str] = field(default_factory=list)
    must_call_tools: list[str] = field(default_factory=list)
    soft_checks: list[dict] = field(default_factory=list)
    optional: bool = False  # reported, not counted in pass rates
    # Hold-out cases were never used to tune the prompt. They are scored separately, so the
    # main pass rate can be compared with how well the prompt generalises.
    holdout: bool = False


def load_cases(path: Path) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = [Case(**item) for item in raw]
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids: {sorted(duplicates)}")
    return cases


# ---------- text matching ----------

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def _normalise(text: str) -> str:
    """Lowercase, unify apostrophes/dashes, drop thousands separators (18,000 -> 18000)."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[‐‑–—]", "-", text)
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def _date_pattern(needle: str) -> str | None:
    """'19 November 2026' -> regex for '19 November 2026' / '19 Nov 2026' / '2026-11-19' / 'November 19, 2026'."""
    match = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", needle)
    if not match or match.group(2) not in MONTHS:
        return None
    day, month_name, year = int(match.group(1)), match.group(2), match.group(3)
    month = rf"(?:{month_name}|{month_name[:3]}\.?)"
    ordinal = r"(?:st|nd|rd|th)?"
    return (
        rf"\b{day}{ordinal}\s+{month},?\s+{year}\b"
        rf"|\b{month}\s+{day}{ordinal},?\s+{year}\b"
        rf"|\b{year}-{MONTHS.index(month_name) + 1:02d}-{day:02d}\b"
    )


def contains(text: str, needle: str) -> bool:
    """Case-insensitive match. A number only matches a whole number ("20" is not in "2026")."""
    text, needle = _normalise(text), _normalise(needle.strip())
    pattern = _date_pattern(needle)
    if pattern is None:
        pattern = re.escape(needle)
        if needle[:1].isdigit():
            pattern = r"(?<![\d.])" + pattern
        if needle[-1:].isdigit():
            pattern += r"(?:\.0+)?(?!\d|\.\d)"  # "18000" also matches "18000.00"
    return re.search(pattern, text) is not None


def contains_any(text: str, needle: str | list[str]) -> bool:
    options = needle if isinstance(needle, list) else [needle]
    return any(contains(text, option) for option in options)


def is_related_section(expected: str, cited: str) -> bool:
    """Same section, or one is a parent/child of the other ("4.1" ~ "4.1(d)")."""
    def under(child: str, parent: str) -> bool:
        return child.startswith(parent + "(") or child.startswith(parent + ".")

    return expected == cited or under(cited, expected) or under(expected, cited)


# ---------- checking one case ----------

@dataclass
class CaseResult:
    case: Case
    passed: bool
    reasons: list[str]  # why it failed (empty if passed)
    warnings: list[str]  # soft checks: reported, never counted
    response: AssistantResponse | None = None
    error: str | None = None
    provenance_failed: bool = False


def check_case(case: Case, response: AssistantResponse) -> CaseResult:
    reasons: list[str] = []
    text = response.text

    if response.type != case.expected_type:
        reasons.append(f"type {response.type!r}, expected {case.expected_type!r}")

    # Exact texts for the fixed templates.
    if case.expected_type == "not_covered" and text != FALLBACK_TEXT:
        reasons.append("text is not exactly FALLBACK_TEXT")
    if case.expected_type == "privacy_refusal" and not text.startswith(PRIVACY_REFUSAL_TEXT):
        reasons.append("text does not start with the exact PRIVACY_REFUSAL_TEXT")
    if case.expected_type == "partial" and FALLBACK_TEXT not in text:
        reasons.append("FALLBACK_TEXT missing from partial answer")

    if case.expected_sections:
        cited = [c.section_id for c in response.citations]
        if not any(is_related_section(e, c) for e in case.expected_sections for c in cited):
            reasons.append(f"cited {cited or 'nothing'}, expected any of {case.expected_sections}")

    for needle in case.must_include:
        if not contains_any(text, needle):
            reasons.append(f"missing {needle!r}")
    for needle in case.must_not_include:
        if contains_any(text, needle):
            reasons.append(f"should not include {needle!r}")

    called = {t["name"] for t in response.debug.get("tool_calls", [])}
    for tool in case.must_call_tools:
        if tool not in called:
            reasons.append(f"tool {tool} not called")

    # §9.5: a number in the answer that comes from nowhere counts as a failure in evals.
    unverified = response.debug.get("unverified_numbers") or []
    if unverified:
        reasons.append(f"unverified numbers {unverified}")

    warnings = []
    for soft in case.soft_checks:
        if contains(text, soft["if_includes"]) and not any(contains(text, r) for r in soft["require_any"]):
            warnings.append(f"says {soft['if_includes']!r} without any of {soft['require_any']}")

    return CaseResult(case, not reasons, reasons, warnings, response, provenance_failed=bool(unverified))


# ---------- running ----------

def run_case(case: Case, provider: Provider, policy: Policy, busy_wait_s: float = BUSY_WAIT_S) -> CaseResult:
    for attempt in range(BUSY_RETRIES + 1):
        try:
            response = ask(case.question, today=EVAL_TODAY, provider=provider, policy=policy)
            return check_case(case, response)
        except ProviderBusyError as e:
            if attempt == BUSY_RETRIES:
                return CaseResult(case, False, ["provider busy"], [], error=str(e))
            print(f"    busy, waiting {busy_wait_s:.0f}s ...", flush=True)
            time.sleep(busy_wait_s)
        except ProviderError as e:
            return CaseResult(case, False, ["provider error"], [], error=str(e))


def run_cases(cases: list[Case], provider: Provider, policy: Policy, delay_s: float) -> list[CaseResult]:
    results: list[CaseResult] = []
    consecutive_busy = 0
    for i, case in enumerate(cases):
        if i:
            time.sleep(delay_s)
        result = run_case(case, provider, policy)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{i + 1}/{len(cases)}] {case.id} {status} {'; '.join(result.reasons)}", flush=True)

        consecutive_busy = consecutive_busy + 1 if result.reasons == ["provider busy"] else 0
        if consecutive_busy >= MAX_CONSECUTIVE_BUSY:
            print(f"Stopping: {MAX_CONSECUTIVE_BUSY} cases in a row failed with the provider busy.")
            break
    return results


# ---------- scoring and report ----------

@dataclass
class Score:
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def __str__(self) -> str:
        return f"{self.passed}/{self.total} ({self.rate:.0%})"


def score(results: list[CaseResult], holdout: bool = False) -> tuple[dict[str, Score], Score]:
    """Pass rates for the main set (default) or the hold-out set. Optional cases never count."""
    counted = [r for r in results if not r.case.optional and r.case.holdout == holdout]
    by_category: dict[str, Score] = {}
    for r in counted:
        s = by_category.setdefault(r.case.category, Score(0, 0))
        s.total += 1
        s.passed += r.passed
    overall = Score(sum(r.passed for r in counted), len(counted))
    return by_category, overall


def targets_met(results: list[CaseResult]) -> tuple[bool, list[str]]:
    """Check §12.4 on the main set (hold-out cases are reported, not targeted). Returns (all met, misses)."""
    by_category, overall = score(results)
    misses = [
        f"{category} {by_category[category]} < {target:.0%}"
        for category, target in CATEGORY_TARGETS.items()
        if category in by_category and by_category[category].rate < target
    ]
    provenance = [
        r.case.id for r in results
        if r.case.category == "calculation" and r.provenance_failed and not r.case.holdout
    ]
    if provenance:
        misses.append(f"calculation cases with unverified numbers: {provenance}")
    if overall.rate < OVERALL_TARGET:
        misses.append(f"overall {overall} < {OVERALL_TARGET:.0%}")
    return not misses, misses


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def provider_settings(provider: Provider) -> str:
    """Model order and thinking level, so a report says exactly what produced it."""
    models = getattr(provider, "models", [provider.model])
    thinking = getattr(provider, "thinking_level", None) or "model default"
    return f"{provider.name}: models {' → '.join(models)}; thinking level {thinking}"


def write_report(
    results: list[CaseResult],
    label: str,
    settings: str,
    total_cases: int,
    started: datetime,
    path: Path,
) -> None:
    by_category, overall = score(results)
    holdout_by_category, holdout_overall = score(results, holdout=True)
    met, misses = targets_met(results)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()[:8]
    models = sorted({m for r in results if r.response for m in r.response.debug.get("models_used", []) if m})
    partial_run = len(results) < total_cases

    lines = [
        f"# Eval results — {label}",
        "",
        f"- Run: {started:%Y-%m-%d %H:%M} (Asia/Dhaka); eval date {EVAL_TODAY.isoformat()}",
        f"- Settings: {settings}",
        f"- Models that answered: {', '.join(models) or '—'}",
        f"- System prompt: `prompts/system_prompt.md` sha256 `{prompt_hash}`",
        f"- Cases run: {len(results)} of {total_cases}" + (" (**partial run**)" if partial_run else ""),
        f"- **Overall: {overall}** — targets {'met' if met else 'NOT met'}"
        + (" for the cases run" if partial_run else ""),
    ]
    lines += [f"  - {m}" for m in misses]
    if holdout_overall.total:
        lines.append(f"- **Hold-out (never used for prompt tuning): {holdout_overall}**")
    lines += ["", "## By category", "", "| Category | Passed | Target |", "|---|---|---|"]
    for category, s in sorted(by_category.items()):
        target = CATEGORY_TARGETS.get(category)
        lines.append(f"| {category} | {s} | {f'{target:.0%}' if target else '—'} |")
    lines.append(f"| **overall** | **{overall}** | {OVERALL_TARGET:.0%} |")
    if holdout_overall.total:
        lines += ["", "## Hold-out by category", "", "| Category | Passed |", "|---|---|"]
        lines += [f"| {category} | {s} |" for category, s in sorted(holdout_by_category.items())]
        lines.append(f"| **hold-out overall** | **{holdout_overall}** |")

    lines += ["", "## Cases", "", "| Id | Category | Result | Type | Reason | Latency | Models |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        status = "pass" if r.passed else "**FAIL**"
        if r.case.optional:
            status += " (optional)"
        if r.case.holdout:
            status += " (hold-out)"
        debug = r.response.debug if r.response else {}
        reason = "; ".join(r.reasons + [f"warning: {w}" for w in r.warnings])
        lines.append(
            f"| {r.case.id} | {r.case.category} | {status} | {r.response.type if r.response else '—'} "
            f"| {_cell(reason)} | {debug.get('latency_s', '—')} | {', '.join(m for m in debug.get('models_used', []) if m)} |"
        )

    failed = [r for r in results if not r.passed or r.warnings]
    if failed:
        lines += ["", "## Details of failures and warnings"]
    for r in failed:
        lines += ["", f"### {r.case.id} — {r.case.question}", ""]
        if r.error:
            lines.append(f"Error: {r.error}")
            continue
        debug = r.response.debug
        lines += ["```text", r.response.text, "```", ""]
        lines.append(f"- Guardrail reason: {debug.get('reason', '—')}; model type: {debug.get('model_type')}")
        for c in debug.get("citation_checks", []):
            lines.append(f"- Citation {c['section_id']} ({'ok' if c['valid'] else 'rejected'}): \"{c['quote']}\"")
        for t in debug.get("tool_calls", []):
            lines.append(f"- Tool {t['name']}({t['args']}) -> {t['result']}")
        if r.response.calculations:
            lines.append(f"- Calculations: {r.response.calculations}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- CLI ----------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles may not default to UTF-8

    parser = argparse.ArgumentParser(prog="python -m evals.run_evals")
    parser.add_argument("--provider", choices=["gemini", "openai_compat"])
    parser.add_argument("--model", help="primary model (fallback models are still tried when it is busy)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="use only the primary provider and model, so the results measure that one model")
    parser.add_argument("--only", help="comma-separated case ids, e.g. A01,B03")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="seconds between cases")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--policy", type=Path, help="policy file (default: POLICY_JSON / POLICY_PATH)")
    args = parser.parse_args()

    settings = load_settings()
    if args.provider:
        settings = dataclasses.replace(settings, llm_provider=args.provider)
    if args.model:
        field_name = "gemini_model" if settings.llm_provider == "gemini" else "openai_compat_model"
        settings = dataclasses.replace(settings, **{field_name: args.model})
    if args.policy:
        settings = dataclasses.replace(settings, policy_json=None, policy_path=args.policy.resolve())

    try:
        cases = load_cases(args.cases)
        policy = load_policy(settings)
        provider = get_provider(settings, fallback=not args.no_fallback)
    except (OSError, ValueError, PolicyNotFoundError, ProviderError) as e:
        sys.exit(f"Error: {e}")

    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        unknown = wanted - {c.id for c in cases}
        if unknown:
            sys.exit(f"Error: unknown case ids {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]

    started = datetime.now(TIMEZONE)
    primary_model = provider.model  # the provider may switch models during the run
    settings_line = provider_settings(provider)
    print(f"Running {len(cases)} cases (eval date {EVAL_TODAY}) — {settings_line}")
    results = run_cases(cases, provider, policy, args.delay)

    report = RESULTS_DIR / f"{provider.name}_{primary_model}_{started:%Y%m%d-%H%M%S}.md"
    write_report(results, f"{provider.name} / {primary_model}", settings_line, len(cases), started, report)

    by_category, overall = score(results)
    holdout_by_category, holdout_overall = score(results, holdout=True)
    met, misses = targets_met(results)
    print("\nBy category:")
    for category, s in sorted(by_category.items()):
        print(f"  {category:<17} {s}")
    print(f"  {'overall':<17} {overall}")
    if holdout_overall.total:
        print(f"Hold-out (never used for prompt tuning): {holdout_overall}")
        for category, s in sorted(holdout_by_category.items()):
            print(f"  {category:<17} {s}")
    partial = " (for the cases run — this was a partial run)" if args.only else ""
    print(f"Targets met{partial}." if met else "Targets NOT met:\n  " + "\n  ".join(misses))
    print(f"Report: {report.relative_to(PROJECT_ROOT)}")
    sys.exit(0 if met and len(results) == len(cases) else 1)


if __name__ == "__main__":
    main()
