# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Simple curve bootstrapping utilities for deposits and par swaps.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._validation import positive_integer
from .calendar import _to_date
from .curves import InterpolatedZeroCurve
from .time import DayCount, year_fraction

__all__ = ["DepositQuote", "SwapQuote", "bootstrap_zero_curve"]


@dataclass(frozen=True)
class DepositQuote:
    """Continuously compounded deposit quote at a calendar maturity."""

    maturity: dt.date
    rate: float  # continuously compounded

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity", _to_date(self.maturity))
        if not math.isfinite(self.rate):
            raise ValueError("deposit rate must be finite")


@dataclass(frozen=True)
class SwapQuote:
    """Par swap quote on the simplified regular accrual grid."""

    maturity: dt.date
    rate: float  # par rate with annualized convention
    payments_per_year: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity", _to_date(self.maturity))
        positive_integer(self.payments_per_year, "payments_per_year")
        if not math.isfinite(self.rate):
            raise ValueError("swap rate must be finite")


def _key(time: float) -> float:
    return round(time, 10)


def bootstrap_zero_curve(
    pricing_date: dt.date,
    deposits: Sequence[DepositQuote],
    swaps: Sequence[SwapQuote],
    *,
    day_count: DayCount | float=DayCount.ACT_365,
) -> InterpolatedZeroCurve:
    """
    Bootstrap an `InterpolatedZeroCurve` from deposit and swap quotes.
    """

    if not deposits and not swaps:
        raise ValueError("Provide at least one deposit or swap quote")

    pricing_date = _to_date(pricing_date)
    maturities = [q.maturity for q in deposits] + [q.maturity for q in swaps]
    if len(set(maturities)) != len(maturities):
        raise ValueError("duplicate quote maturities")
    df_map: dict[float, float] = {}
    node_map: dict[dt.date, float] = {}

    def add_node(maturity: dt.date, t: float, df: float) -> None:
        if not math.isfinite(df) or df <= 0:
            raise ValueError("bootstrapped discount factors must be positive")
        key = _key(t)
        df_map[key] = df
        zero = -math.log(df) / t
        node_map[maturity] = zero

    for deposit in sorted(deposits, key=lambda q: q.maturity):
        t = year_fraction(pricing_date, deposit.maturity, day_count)
        if t <= 0:
            raise ValueError("Deposit maturity must be after pricing date")
        df = math.exp(-deposit.rate * t)
        add_node(deposit.maturity, t, df)

    for swap in sorted(swaps, key=lambda q: q.maturity):
        tau = 1.0 / swap.payments_per_year
        t = year_fraction(pricing_date, swap.maturity, day_count)
        if t <= 0:
            raise ValueError("Swap maturity must be after pricing date")
        periods = int(round(t * swap.payments_per_year))
        if not math.isclose(t, periods * tau, abs_tol=1e-8):
            raise ValueError("swap maturity must align to regular accrual grid")
        if periods < 1:
            raise ValueError("Swap maturity shorter than a coupon period")
        accrual_sum = 0.0
        for k in range(1, periods):
            time_point = _key(k * tau)
            if time_point not in df_map:
                raise ValueError(
                    "Missing discount factor for tenor %.6f" % (k * tau)
                )
            accrual_sum += tau * df_map[time_point]
        numerator = 1.0 - swap.rate * accrual_sum
        denominator = 1.0 + swap.rate * tau
        if denominator <= 0:
            raise ValueError("positive terminal accumulation required")
        df_t = numerator / denominator
        add_node(swap.maturity, t, df_t)

    nodes = sorted(node_map.items(), key=lambda item: item[0])
    return InterpolatedZeroCurve(
        name="bootstrapped",
        reference_date=pricing_date,
        nodes=nodes,
        day_count=day_count,
    )
