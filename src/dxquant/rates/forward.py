# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Multi-curve rate bootstrapping: forward (projection) curve from par swaps.

Given a discount curve (e.g. bootstrapped from OIS) and a sequence of
co-starting
par IRS quotes with increasing maturity and identical frequency, the forward
curve
is bootstrapped sequentially: each swap adds one unknown period forward rate at
the
long end, solved from the par-rate condition.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..bootstrapping import SwapQuote
from ..calendar import _to_date
from ..curves import ConstantShortRateCurve, InterpolatedZeroCurve
from ..time import DayCount, year_fraction

__all__ = ["ForwardCurve", "bootstrap_forward_curve"]


@dataclass
class ForwardCurve:
    """
    Period forward rates F_i over intervals ``[T_{i-1}, T_i]``.

    Use :func:`bootstrap_forward_curve` to construct instances; the forward rate
    for a period is the simply-compounded rate that makes a FRA over that period
    fair given the discount curve.
    """

    period_starts: list[dt.date]
    period_ends: list[dt.date]
    forward_rates: np.ndarray
    day_count: DayCount | float

    def __post_init__(self) -> None:
        self.period_starts = [_to_date(date) for date in self.period_starts]
        self.period_ends = [_to_date(date) for date in self.period_ends]
        self.forward_rates = np.asarray(self.forward_rates, dtype=float).copy()
        if self.forward_rates.ndim != 1 or not self.forward_rates.size:
            raise ValueError("forward rates must be a nonempty vector")
        if not (
            len(self.period_starts)
            == len(self.period_ends)
            == len(self.forward_rates)
        ):
            raise ValueError(
                "period_starts, period_ends, forward_rates must align"
            )
        if self.forward_rates.ndim != 1 or not len(self.forward_rates):
            raise ValueError("forward rates must be a nonempty vector")
        if not np.all(np.isfinite(self.forward_rates)):
            raise ValueError("forward rates must be finite")
        previous = self.period_starts[0]
        for start, end, rate in zip(
            self.period_starts,
            self.period_ends,
            self.forward_rates,
            strict=True,
        ):
            if start != previous:
                raise ValueError("forward periods must be contiguous")
            tau = year_fraction(start, end, self.day_count)
            if tau <= 0 or 1 + rate * tau <= 0:
                raise ValueError(
                    "positive accrual/accumulation factors required"
                )
            previous = end

    def rate_for_period(self, index: int) -> float:
        """Return a stored simply compounded period forward."""

        return float(self.forward_rates[index])

    def _log_accumulation(self, target: dt.date) -> float:
        origin = self.period_starts[0]
        dates = [origin] + self.period_ends
        times = [(date - origin).days for date in dates]
        logs = [0.0]
        for start, end, rate in zip(
            self.period_starts,
            self.period_ends,
            self.forward_rates,
            strict=True,
        ):
            logs.append(
                logs[-1]
                + math.log1p(rate * year_fraction(start, end, self.day_count))
            )
        return float(
            np.interp(
                (target - origin).days, times, logs
            )
        )

    def forward_rate(self, start: dt.date, end: dt.date) -> float:
        """Simple forward from piecewise log-linear accumulation factors.

        Stored periods reprice exactly. Covered subintervals compound the
        implied constant calendar-time rate within each period. Extrapolation
        and nonpositive accrual intervals are rejected.
        """

        start, end = _to_date(start), _to_date(end)
        tau = year_fraction(start, end, self.day_count)
        if tau <= 0:
            raise ValueError("end must be after start")
        if start < self.period_starts[0] or end > self.period_ends[-1]:
            raise ValueError("forward interval outside curve domain")
        log_factor = self._log_accumulation(end) - self._log_accumulation(start)
        return math.expm1(log_factor) / tau


def bootstrap_forward_curve(
    discount_curve: ConstantShortRateCurve | InterpolatedZeroCurve,
    pricing_date: dt.date,
    swaps: Sequence[SwapQuote],
    *,
    day_count: DayCount | float=DayCount.ACT_365,
) -> ForwardCurve:
    """
    Bootstrap a :class:`ForwardCurve` from co-starting par swaps.

    ``swaps`` must be sorted by maturity, share the same ``payments_per_year``,
    and include a quote at every regular coupon date, anchored to pricing_date.
    Sparse, duplicate, off-grid, and stub quotes are rejected. No missing
    projection rates are inferred from the discount curve.
    """

    pricing_date = _to_date(pricing_date)
    if not swaps:
        raise ValueError("Provide at least one swap quote")
    swaps_sorted = sorted(swaps, key=lambda q: q.maturity)
    ppy = swaps_sorted[0].payments_per_year
    if ppy not in (1, 2, 3, 4, 6, 12):
        raise ValueError("payments_per_year must divide 12")
    if any(q.payments_per_year != ppy for q in swaps_sorted):
        raise ValueError("All swaps must share the same payments_per_year")
    maturities = [q.maturity for q in swaps_sorted]
    if len(set(maturities)) != len(maturities):
        raise ValueError("duplicate swap maturities")
    if any(
        q.maturity <= pricing_date or not math.isfinite(q.rate)
        for q in swaps_sorted
    ):
        raise ValueError("future maturities and finite par rates required")
    months = 12 // ppy
    dates = [pricing_date]
    terminal = maturities[-1]
    for index in range(1, len(swaps_sorted) + 1):
        expected = (
            pd.Timestamp(pricing_date) + pd.DateOffset(months=months * index)
        ).date()
        dates.append(expected)
    if dates[-1] != terminal or dates[1:] != maturities:
        raise ValueError("one par quote at every regular coupon date required")
    forwards: list[float] = []
    running_float = 0.0
    annuity = 0.0
    for index, quote in enumerate(swaps_sorted):
        start, end = dates[index : index + 2]
        tau = year_fraction(start, end, day_count)
        df = discount_curve.discount_factor(end)
        if tau <= 0 or not math.isfinite(df) or df <= 0:
            raise ValueError(
                "positive finite accruals/discount factors required"
            )
        annuity += tau * df
        forward = (quote.rate * annuity - running_float) / (tau * df)
        forwards.append(forward)
        running_float += forward * tau * df
    return ForwardCurve(dates[:-1], dates[1:], np.asarray(forwards), day_count)
