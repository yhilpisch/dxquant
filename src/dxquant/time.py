# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Calendar-date clocks shared by valuation and simulation.

ACT/365 is the default; named DayCount conventions and finite positive
numeric denominators are supported throughout scalar/vector fractions,
curves, swaps, and the configurable Monte Carlo clock. Datetime clock times
are discarded for year fractions. Scalar intervals reject reversed dates;
the vector helper deliberately sorts and anchors at the earliest date.
The datetime-array normalizer separately preserves input clock times.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import overload

import numpy as np

from ._validation import positive_integer
from .calendar import _to_date

__all__ = [
    "DayCount",
    "ensure_datetime_array",
    "year_fraction",
    "year_fractions",
    "uniform_time_grid",
]


class DayCount(Enum):
    """
    Named day-count conventions backed by a year-fraction implementation.

    ``ACT_365`` is the package default. ``THIRTY_360`` uses the bond-basis
    formula
    ``(360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360``. Values are unique string
    tags so
    the members do not collapse into Enum aliases (they share the denominator
    360.0).
    """

    ACT_365 = "ACT/365"
    ACT_360 = "ACT/360"
    ACT_365_25 = "ACT/365.25"
    THIRTY_360 = "30/360"

    @property
    def denominator(self) -> float:
        """
        Denominator used by the actual/N conventions (360.0 for 30/360 too).
        """

        return {
            DayCount.ACT_365: 365.0,
            DayCount.ACT_360: 360.0,
            DayCount.ACT_365_25: 365.25,
            DayCount.THIRTY_360: 360.0,
        }[self]


@overload
def ensure_datetime_array(times: Sequence[dt.datetime]) -> np.ndarray: ...


@overload
def ensure_datetime_array(times: Sequence[dt.date]) -> np.ndarray: ...


def ensure_datetime_array(times: Sequence[dt.date | dt.datetime]) -> np.ndarray:
    """
    Convert any iterable of ``datetime``/``date`` objects into a sorted
    ``datetime`` NumPy array.

    Parameters
    ----------
    times:
        Iterable containing ``datetime`` or ``date`` instances describing the
        time grid.

    Returns
    -------
    ndarray
        Array of ``datetime`` objects sorted in ascending order so downstream
        calculations can
        rely on deterministic ordering.
    """

    if not isinstance(times, Iterable):
        raise TypeError("times must be an iterable of datetime/date objects")

    normalized: list[dt.datetime] = []
    for value in times:
        if isinstance(value, dt.datetime):
            normalized.append(value)  # preserve datetime entries as-is
        elif isinstance(value, dt.date):
            normalized.append(
                dt.datetime.combine(value, dt.time())
            )  # promote dates to datetimes
        else:  # pragma: no cover - defensive guard for misuse
            msg = (
                f"Unsupported time entry {value!r} "
                f"(type {type(value).__name__})"
            )
            raise TypeError(msg)

    if not normalized:
        raise ValueError("times iterable is empty")

    array = np.array(
        sorted(normalized)
    )  # sort to guarantee non-decreasing grid
    return array


def year_fractions(
    times: Sequence[dt.date | dt.datetime],
    day_count: DayCount | float=DayCount.ACT_365,
) -> np.ndarray:
    """Calendar-date fractions, sorted and anchored at the earliest date.

    Accepts the same convention/denominator as year_fraction. Datetime clock
    times do not contribute fractional days. Use year_fraction for ordered
    intervals; this vector helper deliberately preserves its sorting contract.
    """

    ordered = sorted(_to_date(value) for value in times)
    if not ordered:
        raise ValueError("times iterable is empty")
    return np.array(
        [year_fraction(ordered[0], value, day_count) for value in ordered]
    )


def uniform_time_grid(maturity: float, num_steps: int) -> np.ndarray:
    """
    Build an evenly spaced time grid from 0 to ``maturity`` (inclusive).
    """

    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError("maturity must be positive")
    positive_integer(num_steps, "num_steps")
    grid = np.linspace(0.0, maturity, num_steps + 1)  # include start and end
    return grid


def _thirty_360(start: dt.date, end: dt.date) -> float:
    """Bond-basis year fraction: (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360."""

    d1 = min(start.day, 30)
    d2 = end.day
    if d1 == 30 and d2 == 31:
        d2 = 30  # end-of-month adjustment
    return (
        360.0 * (end.year - start.year)
        + 30.0 * (end.month - start.month)
        + (d2 - d1)
    ) / 360.0


def year_fraction(
    start: dt.date | dt.datetime,
    end: dt.date | dt.datetime,
    convention: DayCount | float=DayCount.ACT_365,
) -> float:
    """
    Year fraction between two dates under a given day-count convention.

    Parameters
    ----------
    start, end:
        Boundary dates; ``datetime`` inputs are promoted to ``date``.
    convention:
        A :class:`DayCount` member or a raw denominator (``365.0``, ``360.0``,
        ...).
        ``DayCount.THIRTY_360`` selects the 30/360 bond-basis formula.
    """

    start_d = _to_date(start)
    end_d = _to_date(end)
    if end_d < start_d:
        raise ValueError("end must not precede start")

    if isinstance(convention, DayCount) and convention is DayCount.THIRTY_360:
        return _thirty_360(start_d, end_d)

    denominator = (
        convention.denominator
        if isinstance(convention, DayCount)
        else float(convention)
    )
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("day_count denominator must be strictly positive")
    return (end_d - start_d).days / denominator
