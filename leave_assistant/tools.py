"""Deterministic calculation tools the model can call (PRD §8.4).

The model decides which policy rule applies; these functions do the arithmetic and
calendar maths. They are pure functions: no I/O, and invalid input returns
{"error": "..."} instead of raising, so a bad tool call never crashes the tool loop.

Dates are ISO strings (YYYY-MM-DD) in and out. Outputs also include the weekday and a
display form like "1 November 2026".
"""

import ast
import operator
import re
from datetime import date, timedelta
from typing import Any, Callable

# Section 5.1: the weekly holidays are Friday and Saturday. (Monday = 0 ... Sunday = 6)
WEEKLY_HOLIDAYS = {4: "Friday", 5: "Saturday"}
PUBLIC_HOLIDAY_NOTE = "Public holidays not excluded — list not in policy (5.2)."
MAX_RANGE_DAYS = 800  # guard against absurd ranges in day-by-day counting
MAX_EXPRESSION_LENGTH = 200


class ToolInputError(ValueError):
    pass


# ---------- helpers ----------

def _parse_date(value: Any, name: str = "date") -> date:
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise ToolInputError(f"{name} {value!r} is not a valid date in YYYY-MM-DD format")


def _parse_int(value: Any, name: str) -> int:
    # Models sometimes send 2.0 or "2" for an integer argument.
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{name} must be a whole number")
    if not number.is_integer():
        raise ToolInputError(f"{name} must be a whole number")
    return int(number)


def display_date(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def _date_result(d: date) -> dict:
    return {"date": d.isoformat(), "weekday": f"{d:%A}", "display": display_date(d)}


def _is_working_day(d: date) -> bool:
    return d.weekday() not in WEEKLY_HOLIDAYS


def _clean_number(value: float) -> int | float:
    value = round(value, 6)
    return int(value) if float(value).is_integer() else value


# ---------- tools ----------

_OPERATORS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolInputError("only numbers, + - * / and brackets are allowed")


def calculate(expression: str) -> dict:
    """Arithmetic only. Parsed with ast — never eval()."""
    if not isinstance(expression, str) or not expression.strip():
        raise ToolInputError("expression must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolInputError("expression is too long")
    # Accept thousands separators ("45,000") and common symbols for multiply/divide.
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression)
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        raise ToolInputError(f"could not parse expression {expression!r}")
    try:
        value = _evaluate(tree)
    except ZeroDivisionError:
        raise ToolInputError("division by zero")
    return {"expression": expression, "result": _clean_number(value)}


def add_calendar_days(date: str, days: int) -> dict:
    start, n = _parse_date(date), _parse_int(days, "days")
    return _date_result(start + timedelta(days=n))


def add_weeks(date: str, weeks: int) -> dict:
    start, n = _parse_date(date), _parse_int(weeks, "weeks")
    return _date_result(start + timedelta(weeks=n))


def add_working_days(date: str, days: int) -> dict:
    """Move forward (or back, if negative) by N working days, skipping Friday and Saturday."""
    current, n = _parse_date(date), _parse_int(days, "days")
    if abs(n) > MAX_RANGE_DAYS:
        raise ToolInputError("days is too large")
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining:
        current += timedelta(days=step)
        if _is_working_day(current):
            remaining -= 1
    return _date_result(current)


def _date_range(start: str, end: str) -> list[date]:
    first, last = _parse_date(start, "start"), _parse_date(end, "end")
    if last < first:
        raise ToolInputError("end must be on or after start")
    if (last - first).days > MAX_RANGE_DAYS:
        raise ToolInputError("date range is too long")
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def count_working_days(start: str, end: str) -> dict:
    """Working days from start to end inclusive, excluding Friday and Saturday."""
    days = _date_range(start, end)
    working = [d for d in days if _is_working_day(d)]
    excluded = [d for d in days if not _is_working_day(d)]
    return {
        "count": len(working),
        "dates": [f"{d.isoformat()} ({d:%a})" for d in working],
        "excluded_weekly_holidays": [f"{d.isoformat()} ({d:%A})" for d in excluded],
        "note": PUBLIC_HOLIDAY_NOTE,
    }


def count_calendar_days(start: str, end: str) -> dict:
    """Calendar days from start to end inclusive."""
    return {"count": len(_date_range(start, end))}


def months_between(start: str, end: str) -> dict:
    """Whole months from start to end, plus the days left over."""
    first, last = _parse_date(start, "start"), _parse_date(end, "end")
    if last < first:
        raise ToolInputError("end must be on or after start")
    months = (last.year - first.year) * 12 + (last.month - first.month)
    if last.day < first.day:
        months -= 1
    # Anchor = start moved forward by `months`, clamped to the end of a short month.
    year, month = divmod(first.month - 1 + months, 12)
    year, month = first.year + year, month + 1
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    anchor = date(year, month, min(first.day, (next_month - timedelta(days=1)).day))
    return {"full_months": months, "extra_days": (last - anchor).days}


def weekday_of(date: str) -> dict:
    return _date_result(_parse_date(date))


# ---------- registry and dispatcher ----------

TOOLS: dict[str, Callable[..., dict]] = {
    "calculate": calculate,
    "add_calendar_days": add_calendar_days,
    "add_weeks": add_weeks,
    "add_working_days": add_working_days,
    "count_working_days": count_working_days,
    "count_calendar_days": count_calendar_days,
    "months_between": months_between,
    "weekday_of": weekday_of,
}

_DATE = {"type": "string", "description": "Date in YYYY-MM-DD format"}

# Provider-neutral tool definitions (JSON Schema). Each provider converts these.
TOOL_SPECS: list[dict] = [
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression with numbers, + - * / and brackets. "
        "Use for encashment, Leave Without Pay deductions and leave accrual.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 45000 / 30 * 12"}},
            "required": ["expression"],
        },
    },
    {
        "name": "add_calendar_days",
        "description": "Add (or subtract, if negative) calendar days to a date. "
        "Use for notice periods and day windows counted in calendar days.",
        "parameters": {
            "type": "object",
            "properties": {"date": _DATE, "days": {"type": "integer"}},
            "required": ["date", "days"],
        },
    },
    {
        "name": "add_weeks",
        "description": "Add (or subtract, if negative) whole weeks to a date.",
        "parameters": {
            "type": "object",
            "properties": {"date": _DATE, "weeks": {"type": "integer"}},
            "required": ["date", "weeks"],
        },
    },
    {
        "name": "add_working_days",
        "description": "Move forward (or back, if negative) by a number of working days, "
        "skipping the weekly holidays (Friday and Saturday). Does not know public holidays.",
        "parameters": {
            "type": "object",
            "properties": {"date": _DATE, "days": {"type": "integer"}},
            "required": ["date", "days"],
        },
    },
    {
        "name": "count_working_days",
        "description": "Count working days from start to end inclusive, excluding Friday and "
        "Saturday. Does not know public holidays.",
        "parameters": {
            "type": "object",
            "properties": {"start": _DATE, "end": _DATE},
            "required": ["start", "end"],
        },
    },
    {
        "name": "count_calendar_days",
        "description": "Count calendar days from start to end inclusive.",
        "parameters": {
            "type": "object",
            "properties": {"start": _DATE, "end": _DATE},
            "required": ["start", "end"],
        },
    },
    {
        "name": "months_between",
        "description": "Number of full months (and extra days) from start to end. "
        "Use for length of service.",
        "parameters": {
            "type": "object",
            "properties": {"start": _DATE, "end": _DATE},
            "required": ["start", "end"],
        },
    },
    {
        "name": "weekday_of",
        "description": "Day of the week for a date.",
        "parameters": {"type": "object", "properties": {"date": _DATE}, "required": ["date"]},
    },
]


def dispatch(name: str, args: dict | None) -> dict:
    """Run a tool by name. Never raises: errors come back as {"error": "..."}."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return tool(**(args or {}))
    except ToolInputError as e:
        return {"error": str(e)}
    except TypeError as e:  # missing or unexpected arguments
        return {"error": f"bad arguments for {name}: {e}"}
    except (ValueError, OverflowError) as e:
        return {"error": f"{name} failed: {e}"}
