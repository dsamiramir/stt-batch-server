from __future__ import annotations

from datetime import datetime

import pytest

from app.schedule import (
    WindowSpecError,
    is_open,
    next_open_at,
    parse_windows,
    window_closes_at,
)

DEFAULT = "Mon-Fri 22:00-08:00; Sat 00:00-24:00; Sun 00:00-24:00"

# 2026-08-31 is a Monday.
MON_2100 = datetime(2026, 8, 31, 21, 0)
MON_2300 = datetime(2026, 8, 31, 23, 0)
TUE_0300 = datetime(2026, 9, 1, 3, 0)
TUE_0800 = datetime(2026, 9, 1, 8, 0)
TUE_1400 = datetime(2026, 9, 1, 14, 0)
SAT_1200 = datetime(2026, 9, 5, 12, 0)
SUN_2359 = datetime(2026, 9, 6, 23, 59)
MON_0700 = datetime(2026, 9, 7, 7, 0)


@pytest.mark.parametrize(
    "moment,expected",
    [
        (MON_2100, False),  # weekday evening, before the window opens
        (MON_2300, True),   # inside Monday night
        (TUE_0300, True),   # same window, after midnight
        (TUE_0800, False),  # window ends exactly at 08:00
        (TUE_1400, False),  # weekday working hours
        (SAT_1200, True),   # all-day weekend window
        (SUN_2359, True),
        # Sunday's window ends at midnight, so Monday morning is not covered.
        (MON_0700, False),
    ],
)
def test_is_open(moment: datetime, expected: bool) -> None:
    assert is_open(parse_windows(DEFAULT), moment) is expected


def test_overnight_window_end() -> None:
    windows = parse_windows(DEFAULT)
    assert window_closes_at(windows, TUE_0300) == datetime(2026, 9, 1, 8, 0)


def test_next_open_from_closed_moment() -> None:
    windows = parse_windows(DEFAULT)
    assert next_open_at(windows, TUE_1400) == datetime(2026, 9, 1, 22, 0)


def test_next_open_returns_now_when_open() -> None:
    windows = parse_windows(DEFAULT)
    assert next_open_at(windows, SAT_1200) == SAT_1200


def test_friday_night_runs_into_saturday() -> None:
    windows = parse_windows(DEFAULT)
    fri_2330 = datetime(2026, 9, 4, 23, 30)
    sat_0300 = datetime(2026, 9, 5, 3, 0)
    assert is_open(windows, fri_2330)
    assert is_open(windows, sat_0300)


def test_day_shorthand_and_wildcard() -> None:
    assert is_open(parse_windows("* 00:00-24:00"), TUE_1400)
    assert is_open(parse_windows("Sat,Sun 09:00-18:00"), SAT_1200)
    assert not is_open(parse_windows("Sat,Sun 09:00-18:00"), TUE_1400)


def test_bare_range_defaults_to_every_day() -> None:
    windows = parse_windows("13:00-15:00")
    assert is_open(windows, TUE_1400)
    assert not is_open(windows, TUE_0300)


@pytest.mark.parametrize("spec", ["Xyz 10:00-11:00", "Mon 10:00", "Mon 25:00-26:00", ""])
def test_rejects_bad_specs(spec: str) -> None:
    with pytest.raises((WindowSpecError, ValueError)):
        parse_windows(spec)
