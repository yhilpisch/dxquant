# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Business-day calendar utilities built on top of NumPy's ``busday`` toolkit.

The original ``dx`` package mixed ad-hoc date handling into larger modules.
DXQuant provides a dedicated calendar abstraction so schedules, curve
construction, and Monte Carlo routines share the same business-day logic.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

__all__ = ["BusinessDayCalendar"]


def _to_date(value: dt.date | dt.datetime) -> dt.date:
    """
    Promote ``datetime`` objects to ``date`` so NumPy receives ``datetime64[D]``
    values.
    """

    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    msg = f"Unsupported date value {value!r}"
    raise TypeError(msg)


def _to_datetime64(value: dt.date | dt.datetime) -> np.datetime64:
    """Convert Python date-like input into ``datetime64[D]``."""

    return np.datetime64(_to_date(value), "D")


@dataclass(frozen=True)
class BusinessDayCalendar:
    """
    Thin wrapper around ``numpy.busday_*`` helpers providing a Pythonic
    interface.

    Parameters
    ----------
    name:
        Human-friendly identifier (e.g., ``"TARGET2"`` or ``"NYC"``).
    weekmask:
        String describing working days.  Defaults to the standard Monday-Friday
        convention.
    holidays:
        Optional iterable of custom holidays (Python ``date``/``datetime``
        objects).
    """

    name: str
    weekmask: str = "Mon Tue Wed Thu Fri"
    holidays: Sequence[dt.date | dt.datetime] | None = field(default=None)

    def __post_init__(self) -> None:
        # Convert holidays to a sorted tuple so hashing/equality behave
        # predictably.
        normalized: tuple[dt.date, ...] = ()
        if self.holidays:
            normalized = tuple(sorted(_to_date(day) for day in self.holidays))
        object.__setattr__(self, "holidays", normalized)

    # Internal helper used by NumPy's busday APIs.
    def _busday_kwargs(self) -> dict[str, np.ndarray | str]:
        payload: dict[str, np.ndarray | str] = {"weekmask": self.weekmask}
        if self.holidays:
            payload["holidays"] = np.array(
                [np.datetime64(day, "D") for day in self.holidays]
            )
        return payload

    def is_business_day(self, value: dt.date | dt.datetime) -> bool:
        """Return ``True`` if the provided date falls on a business day."""

        day = np.array([_to_datetime64(value)], dtype="datetime64[D]")
        result = np.is_busday(  # type: ignore[call-overload]
            day, **self._busday_kwargs()
        )
        return bool(result[0])

    def add_business_days(
        self, start: dt.date | dt.datetime, offset: int
    ) -> dt.date:
        """
        Shift a date by ``offset`` business days (can be negative).
        """

        start_arr = np.array([_to_datetime64(start)], dtype="datetime64[D]")
        shifted = np.busday_offset(
            start_arr,
            offsets=offset,
            roll="forward",
            **self._busday_kwargs(),
        )  # type: ignore[call-overload]
        return cast(dt.date, shifted[0].astype("datetime64[D]").astype(object))

    def business_days_between(
        self,
        start: dt.date | dt.datetime,
        end: dt.date | dt.datetime,
        *,
        inclusive: bool=False,
    ) -> int:
        """
        Count business days between two dates.

        Parameters
        ----------
        start, end:
            Boundary dates.  If ``inclusive`` is ``False`` (default) the
            function mirrors
            ``numpy.busday_count`` semantics and excludes the end date.
        inclusive:
            When ``True`` the end date participates in the count, matching many
            industry conventions.
        """

        start_arr = np.array([_to_datetime64(start)], dtype="datetime64[D]")
        end_arr = np.array([_to_datetime64(end)], dtype="datetime64[D]")
        count = int(
            np.busday_count(  # type: ignore[call-overload]
                start_arr, end_arr, **self._busday_kwargs()
            )[0]
        )
        if inclusive and self.is_business_day(end):
            count += 1  # include end date when it qualifies as a business day
        return count

    def iter_schedule(
        self,
        start: dt.date | dt.datetime,
        end: dt.date | dt.datetime,
        *,
        include_end: bool=True,
    ) -> Iterator[dt.date]:
        """
        Yield business-day dates from ``start`` to ``end``.
        """

        cursor = _to_date(start)
        stop = _to_date(end)

        while cursor < stop:
            if self.is_business_day(cursor):
                # emit Python ``date`` for ergonomic downstream usage
                yield cursor
            cursor += dt.timedelta(days=1)  # walk day by day

        if include_end and cursor == stop and self.is_business_day(cursor):
            yield cursor  # optionally include end date

    def schedule(
        self,
        start: dt.date | dt.datetime,
        end: dt.date | dt.datetime,
        *,
        include_end: bool=True,
    ) -> list[dt.date]:
        """
        Convenience wrapper returning a list of business-day dates between
        ``start`` and ``end``.
        """

        return list(self.iter_schedule(start, end, include_end=include_end))
