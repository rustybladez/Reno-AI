from leave_assistant.tools import (
    add_calendar_days,
    add_weeks,
    add_working_days,
    calculate,
    count_calendar_days,
    count_working_days,
    dispatch,
    months_between,
    weekday_of,
)


# ---------- known answers from PRD §13 ----------

def test_count_working_days_skips_friday_and_saturday():
    result = count_working_days("2026-11-05", "2026-11-12")
    assert result["count"] == 6
    assert result["excluded_weekly_holidays"] == ["2026-11-06 (Friday)", "2026-11-07 (Saturday)"]
    assert "Public holidays not excluded" in result["note"]


def test_add_working_days_skips_weekend():
    assert add_working_days("2026-10-28", 2) == {
        "date": "2026-11-01", "weekday": "Sunday", "display": "1 November 2026",
    }


def test_add_calendar_days():
    assert add_calendar_days("2026-12-16", 60)["date"] == "2027-02-14"


def test_add_weeks_negative():
    assert add_weeks("2027-03-01", -8)["date"] == "2027-01-04"


def test_months_between_full_months():
    assert months_between("2026-01-26", "2026-10-26") == {"full_months": 9, "extra_days": 0}


def test_calculate_encashment():
    assert calculate("45000/30*12")["result"] == 18000


def test_calculate_rejects_code():
    assert "error" in dispatch("calculate", {"expression": "__import__('os')"})


def test_calculate_division_by_zero():
    assert "error" in dispatch("calculate", {"expression": "1/0"})


def test_invalid_date_returns_error_dict():
    assert "error" in dispatch("add_calendar_days", {"date": "2026-02-30", "days": 1})


# ---------- other behaviour ----------

def test_calculate_accepts_thousands_separators_and_decimals():
    assert calculate("45,000 / 30 * 12")["result"] == 18000
    assert calculate("1.67 * 6")["result"] == 10.02


def test_calculate_rejects_powers_and_names():
    assert "error" in dispatch("calculate", {"expression": "2 ** 1000"})
    assert "error" in dispatch("calculate", {"expression": "salary / 30"})


def test_add_working_days_backwards_and_zero():
    assert add_working_days("2026-11-01", -2)["date"] == "2026-10-28"
    assert add_working_days("2026-10-28", 0)["date"] == "2026-10-28"


def test_paternity_window_dates():
    # Born 20 Oct: 30 days -> 19 Nov; a 7-day block ending 19 Nov starts 13 Nov.
    assert add_calendar_days("2026-10-20", 30)["date"] == "2026-11-19"
    assert add_calendar_days("2026-11-19", -6)["date"] == "2026-11-13"
    assert count_calendar_days("2026-11-13", "2026-11-19") == {"count": 7}


def test_months_between_with_extra_days_and_month_end():
    assert months_between("2026-01-31", "2026-03-15") == {"full_months": 1, "extra_days": 15}
    assert months_between("2026-01-31", "2026-02-28") == {"full_months": 0, "extra_days": 28}


def test_end_before_start_is_an_error():
    assert "error" in dispatch("count_working_days", {"start": "2026-11-12", "end": "2026-11-05"})


def test_weekday_of():
    assert weekday_of("2026-10-26")["weekday"] == "Monday"


def test_dispatch_handles_bad_input():
    assert "error" in dispatch("no_such_tool", {})
    assert "error" in dispatch("add_calendar_days", {"date": "2026-01-01"})  # missing days
    assert "error" in dispatch("add_calendar_days", {"date": "2026-01-01", "days": 1.5})
    assert dispatch("add_calendar_days", {"date": "2026-01-01", "days": 2.0})["date"] == "2026-01-03"
