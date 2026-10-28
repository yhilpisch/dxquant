# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Interest rate swaps with optional separate discount/projection curves.

Supports configurable day-count conventions (``dxquant.time.DayCount`` or a raw
denominator) and optional amortizing notionals via ``notional_schedule``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from dxquant.calendar import BusinessDayCalendar, _to_date
from dxquant.curves import ConstantShortRateCurve, InterpolatedZeroCurve
from dxquant.rates.forward import ForwardCurve
from dxquant.time import DayCount, year_fraction

__all__ = ["InterestRateSwap"]


@dataclass
class InterestRateSwap:
    """
    Fixed-for-floating interest rate swap (single currency).

    Parameters
    ----------
    notional:
        Constant notional used when ``notional_schedule`` is ``None``.
    notional_schedule:
        Optional per-period notional vector (amortizing swap). Its length must
        equal the number of payment periods; the last entry covers the final
        period. When provided it overrides ``notional`` for accrual/Floating PV.
    day_count:
        Day-count convention (``DayCount`` member) or a raw denominator.
    """

    name: str
    fixed_rate: float
    notional: float
    effective_date: dt.date
    termination_date: dt.date
    payment_frequency: str  # e.g. "3M", "6M"
    day_count: DayCount | float = DayCount.ACT_365
    calendar: BusinessDayCalendar | None = None
    notional_schedule: Sequence[float] | None = field(default=None)

    def payment_schedule(self) -> Sequence[dt.date]:
        """
        Build anchored month payments, with an optional final stub/calendar
        roll.
        """

        self.effective_date = _to_date(self.effective_date)
        self.termination_date = _to_date(self.termination_date)
        match = re.fullmatch(r"([1-9][0-9]*)M", self.payment_frequency)
        if not match:
            raise ValueError("payment_frequency must be a positive month tenor")
        if self.termination_date <= self.effective_date:
            raise ValueError("termination must follow effective date")
        if not math.isfinite(self.fixed_rate) or not math.isfinite(
            self.notional
        ):
            raise ValueError("rate/notional must be finite")
        if self.notional < 0:
            raise ValueError("notional must be nonnegative")
        months = int(match[1])
        dates: list[dt.date] = []
        index = 1
        while True:
            pay = (
                pd.Timestamp(self.effective_date)
                + pd.DateOffset(months=months * index)
            ).date()
            if pay >= self.termination_date:
                dates.append(self.termination_date)
                break
            dates.append(pay)
            index += 1
        if self.calendar:
            dates = [self.calendar.add_business_days(d, 0) for d in dates]
        previous = self.effective_date
        for pay in dates:
            if pay <= previous:
                raise ValueError(
                    "adjusted payments must be strictly increasing"
                )
            previous = pay
        if self.notional_schedule is not None:
            if len(self.notional_schedule) != len(dates):
                raise ValueError("notional schedule must match payment periods")
            if not all(
                math.isfinite(n) and n >= 0 for n in self.notional_schedule
            ):
                raise ValueError("notionals must be finite and nonnegative")
        return dates

    def _accrual(self, start: dt.date, end: dt.date) -> float:
        return year_fraction(start, end, self.day_count)

    def _period_notional(self, index: int) -> float:
        if self.notional_schedule is None:
            return self.notional
        return float(self.notional_schedule[index])

    def _legs(
        self,
        discount_curve: ConstantShortRateCurve | InterpolatedZeroCurve,
        projection_curve: ForwardCurve | None,
    ) -> tuple[float, float]:
        previous = self.effective_date
        floating, annuity = 0.0, 0.0
        for index, payment in enumerate(self.payment_schedule()):
            tau = self._accrual(previous, payment)
            if tau <= 0:
                raise ValueError("positive accrual periods required")
            discount = discount_curve.discount_factor(payment)
            notional = self._period_notional(index)
            if projection_curve is None:
                floating += notional * (
                    discount_curve.discount_factor(previous) - discount
                )
            else:
                forward = projection_curve.forward_rate(previous, payment)
                projection_tau = year_fraction(
                    previous, payment, projection_curve.day_count
                )
                floating += notional * projection_tau * forward * discount
            annuity += notional * tau * discount
            previous = payment
        return floating, annuity

    def par_rate(
        self,
        discount_curve: ConstantShortRateCurve | InterpolatedZeroCurve,
        projection_curve: ForwardCurve | None=None,
    ) -> float:
        """Par fixed rate, optionally using separate projection forwards."""

        floating, annuity = self._legs(discount_curve, projection_curve)
        if annuity <= 0:
            raise ValueError("positive fixed-leg annuity required")
        return floating / annuity

    def present_value(
        self,
        discount_curve: ConstantShortRateCurve | InterpolatedZeroCurve,
        projection_curve: ForwardCurve | None=None,
    ) -> float:
        """Payer-swap value: projected floating leg less fixed leg."""

        floating, annuity = self._legs(discount_curve, projection_curve)
        return floating - self.fixed_rate * annuity
