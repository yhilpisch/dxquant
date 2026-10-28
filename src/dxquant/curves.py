# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Yield/discount curve abstractions leveraging DXQuant utilities.

These are modern counterparts to the ``constant_short_rate`` and
``deterministic_short_rate``
structures found in the legacy ``dx`` package.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .calendar import _to_date
from .interpolate import Spline1D
from .time import DayCount, year_fraction

__all__ = ["ConstantShortRateCurve", "InterpolatedZeroCurve"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConstantShortRateCurve:
    """
    Exponential discount curve with a single continuously compounded rate.
    """

    name: str
    reference_date: dt.date | dt.datetime
    rate: float  # continuous compounding
    day_count: DayCount | float = DayCount.ACT_365

    def __post_init__(self) -> None:
        if not np.isfinite(self.rate):
            raise ValueError("rate must be finite")
        year_fraction(self.reference_date, self.reference_date, self.day_count)
        if self.rate < 0:
            raise ValueError(
                "Short rate must be non-negative for this simplified curve"
            )

    def year_fraction(self, target: dt.date | dt.datetime) -> float:
        """Return the ordered fraction under the configured convention."""

        return year_fraction(self.reference_date, target, self.day_count)

    def discount_factor(self, target: dt.date | dt.datetime) -> float:
        """Continuous compounding discount factor."""

        delta = self.year_fraction(target)
        return float(np.exp(-self.rate * delta))  # exp(-r * t)

    def forward_rate(
        self, start: dt.date | dt.datetime, end: dt.date | dt.datetime
    ) -> float:
        """Forward rate over [start, end] using the constant short rate."""

        start_df = self.discount_factor(start)
        end_df = self.discount_factor(end)
        delta = self.year_fraction(end) - self.year_fraction(start)
        if delta <= 0:
            raise ValueError("Forward interval must be positive")
        return float(-np.log(end_df / start_df) / delta)


@dataclass(frozen=True)
class InterpolatedZeroCurve:
    """
    Curve built from date/yield nodes with spline interpolation on zero rates.
    """

    name: str
    reference_date: dt.date | dt.datetime
    nodes: Sequence[tuple[dt.date | dt.datetime, float]]
    day_count: DayCount | float = DayCount.ACT_365
    spline_degree: int = 3
    _spline: Spline1D = field(init=False, repr=False)
    _times: FloatArray = field(init=False, repr=False)
    _yields: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise ValueError("At least two curve nodes are required")
        dates, yields = zip(
            *sorted(self.nodes, key=lambda item: _to_date(item[0])), strict=True
        )

        if any(y < -1.0 for y in yields):
            raise ValueError("Negative yields below -100% unsupported")

        times = np.array(
            [
                year_fraction(self.reference_date, date, self.day_count)
                for date in dates
            ]
        )
        spline = Spline1D(times, yields, degree=self.spline_degree)
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "_times", np.array(times, dtype=float))
        object.__setattr__(self, "_yields", np.array(yields, dtype=float))

    def _time_from_date(self, target: dt.date | dt.datetime) -> float:
        return year_fraction(self.reference_date, target, self.day_count)

    def zero_rate(self, target: dt.date | dt.datetime) -> float:
        """Return the interpolated continuous zero rate."""

        t = self._time_from_date(target)
        return float(self._spline.value([t])[0])

    def discount_factor(self, target: dt.date | dt.datetime) -> float:
        """Continuous compounding discount factor derived from zero rate."""

        if target == self.reference_date:
            return 1.0
        t = self._time_from_date(target)
        r = float(self._spline.value([t])[0])
        return float(np.exp(-r * t))

    def forward_rate(
        self, start: dt.date | dt.datetime, end: dt.date | dt.datetime
    ) -> float:
        """Continuously compounded forward rate from discount factors."""

        start_df = self.discount_factor(start)
        end_df = self.discount_factor(end)
        delta = self._time_from_date(end) - self._time_from_date(start)
        if delta <= 0:
            raise ValueError("Forward interval must be positive")
        return float(-np.log(end_df / start_df) / delta)

    def instantaneous_forward(self, target: dt.date | dt.datetime) -> float:
        """
        Instantaneous forward rate f(t) = r(t) + t * r'(t).
        """

        t = self._time_from_date(target)
        zero = float(self._spline.value([t])[0])
        derivative = float(self._spline.derivative([t])[0])
        return zero + t * derivative
