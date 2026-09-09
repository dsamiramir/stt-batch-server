"""Parsing and evaluation of the off-peak time windows.

A window is written as ``<days> <HH:MM>-<HH:MM>``. The day part is optional and
defaults to every day. Windows whose end is at or before their start wrap past
midnight, so ``Mon-Fri 22:00-08:00`` covers Monday 22:00 through Tuesday 08:00,
and so on through Friday night. ``24:00`` is accepted as end-of-day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

_DAY_INDEX = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_ALL_DAYS = frozenset(range(7))


class WindowSpecError(ValueError):
    """Raised when a window specification cannot be parsed."""


@dataclass(frozen=True)
class Window:
    days: frozenset[int]
    start: time
    duration: timedelta
    raw: str

    def start_on(self, day: date) -> datetime:
        return datetime.combine(day, self.start)

    def __str__(self) -> str:  # pragma: no cover - display only
        days = "every day" if self.days == _ALL_DAYS else ",".join(
            _DAY_NAMES[d] for d in sorted(self.days)
        )
        return f"{days} {self.start:%H:%M} +{self.duration}"


def _parse_days(token: str) -> frozenset[int]:
    token = token.strip().lower()
    if token in ("*", "all", "daily", "everyday", "any"):
        return _ALL_DAYS
    out: set[int] = set()
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (p.strip()[:3] for p in part.split("-", 1))
            if lo not in _DAY_INDEX or hi not in _DAY_INDEX:
                raise WindowSpecError(f"unknown day range: {part!r}")
            cursor = _DAY_INDEX[lo]
            end = _DAY_INDEX[hi]
            while True:
                out.add(cursor)
                if cursor == end:
                    break
                cursor = (cursor + 1) % 7
        else:
            key = part[:3]
            if key not in _DAY_INDEX:
                raise WindowSpecError(f"unknown day: {part!r}")
            out.add(_DAY_INDEX[key])
    if not out:
        raise WindowSpecError(f"no days in {token!r}")
    return frozenset(out)


def _parse_clock(token: str) -> tuple[time, bool]:
    """Return (time, is_end_of_day). ``24:00`` maps to 00:00 + end-of-day flag."""
    token = token.strip()
    if token in ("24:00", "2400", "24"):
        return time(0, 0), True
    try:
        hh, mm = token.split(":")
        parsed = time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise WindowSpecError(f"bad time: {token!r}") from exc
    return parsed, False


def parse_windows(spec: str) -> list[Window]:
    windows: list[Window] = []
    for raw_entry in spec.replace("\n", ";").split(";"):
        entry = raw_entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = entry.rsplit(None, 1)
        if len(parts) == 1:
            days, range_token = _ALL_DAYS, parts[0]
        else:
            days, range_token = _parse_days(parts[0]), parts[1]
        if "-" not in range_token:
            raise WindowSpecError(f"missing time range in {entry!r}")
        start_token, end_token = range_token.split("-", 1)
        start, _ = _parse_clock(start_token)
        end, end_of_day = _parse_clock(end_token)

        anchor = date(2000, 1, 1)
        if end_of_day:
            duration = (
                datetime.combine(anchor + timedelta(days=1), time(0, 0))
                - datetime.combine(anchor, start)
            )
        else:
            duration = datetime.combine(anchor, end) - datetime.combine(anchor, start)
            if duration <= timedelta(0):
                duration += timedelta(days=1)
        windows.append(Window(days=days, start=start, duration=duration, raw=entry))
    if not windows:
        raise WindowSpecError(f"no usable windows in {spec!r}")
    return windows


def active_window(windows: list[Window], now: datetime) -> Window | None:
    """The window currently containing ``now``, if any."""
    for offset in (0, 1):
        day = (now - timedelta(days=offset)).date()
        for window in windows:
            if day.weekday() not in window.days:
                continue
            start = window.start_on(day)
            if start <= now < start + window.duration:
                return window
    return None


def is_open(windows: list[Window], now: datetime) -> bool:
    return active_window(windows, now) is not None


def window_closes_at(windows: list[Window], now: datetime) -> datetime | None:
    """End of the window containing ``now`` (latest end if several overlap)."""
    ends: list[datetime] = []
    for offset in (0, 1):
        day = (now - timedelta(days=offset)).date()
        for window in windows:
            if day.weekday() not in window.days:
                continue
            start = window.start_on(day)
            if start <= now < start + window.duration:
                ends.append(start + window.duration)
    return max(ends) if ends else None


def next_open_at(windows: list[Window], now: datetime) -> datetime | None:
    """Start of the next window after ``now``; ``now`` itself if already open."""
    if is_open(windows, now):
        return now
    candidates: list[datetime] = []
    for offset in range(0, 9):
        day = (now + timedelta(days=offset)).date()
        for window in windows:
            if day.weekday() not in window.days:
                continue
            start = window.start_on(day)
            if start > now:
                candidates.append(start)
    return min(candidates) if candidates else None
