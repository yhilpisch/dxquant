"""Projection accumulation and independent multi-curve par repricing."""

import datetime as dt
import math

import numpy as np
import pytest

from dxquant import (
    ConstantShortRateCurve,
    DayCount,
    ForwardCurve,
    InterestRateSwap,
    SwapQuote,
    bootstrap_forward_curve,
)

ORIGIN = dt.date(2024, 1, 1)
MID = ORIGIN + dt.timedelta(days=180)
END = ORIGIN + dt.timedelta(days=360)


def _curve() -> ForwardCurve:
    return ForwardCurve([ORIGIN, MID], [MID, END], np.array([0.04, 0.08]), 360)


def test_covered_forwards_compound_stored_factors() -> None:
    curve = _curve()
    assert curve.forward_rate(ORIGIN, MID) == pytest.approx(0.04)
    assert curve.forward_rate(MID, END) == pytest.approx(0.08)
    assert curve.forward_rate(ORIGIN, END) == pytest.approx(1.02 * 1.04 - 1)
    quarter = ORIGIN + dt.timedelta(days=90)
    expected = (math.sqrt(1.02) - 1) / 0.25
    assert curve.forward_rate(ORIGIN, quarter) == pytest.approx(expected)
    first = 1 + curve.forward_rate(ORIGIN, quarter) * 0.25
    second = 1 + curve.forward_rate(quarter, END) * 0.75
    assert first * second == pytest.approx(1.02 * 1.04)


@pytest.mark.parametrize(
    "start, end",
    [
        (ORIGIN - dt.timedelta(days=1), MID),
        (MID, END + dt.timedelta(days=1)),
        (MID, MID),
        (END, MID),
    ],
)
def test_forward_domain_is_explicit(start: dt.date, end: dt.date) -> None:
    with pytest.raises(ValueError):
        _curve().forward_rate(start, end)


def test_multicurve_nonflat_quotes_reprice() -> None:
    discount = ConstantShortRateCurve("ois", ORIGIN, 0.01)
    quotes = [
        SwapQuote(ORIGIN.replace(year=2024 + n), r)
        for n, r in [(1, 0.02), (2, 0.027), (3, 0.035)]
    ]
    projection = bootstrap_forward_curve(discount, ORIGIN, quotes)
    for quote in quotes:
        swap = InterestRateSwap(
            "payer", quote.rate, 1_000_000, ORIGIN, quote.maturity, "12M"
        )
        assert abs(swap.present_value(discount, projection)) < 1e-8
        assert swap.par_rate(discount, projection) == pytest.approx(quote.rate)
    assert abs(swap.present_value(discount)) > 1000


def test_different_projection_and_fixed_day_counts() -> None:
    discount = ConstantShortRateCurve("ois", ORIGIN, 0.01)
    swap = InterestRateSwap("payer", 0.02, 1, ORIGIN, END, "6M")
    projection = _curve()
    # Use exact stored period dates to isolate the accrual convention.
    swap.payment_schedule = lambda: [MID, END]  # type: ignore[method-assign]
    fixed = (
        0.02
        * 180
        / 365
        * (discount.discount_factor(MID) + discount.discount_factor(END))
    )
    floating = 0.04 * 0.5 * discount.discount_factor(MID)
    floating += 0.08 * 0.5 * discount.discount_factor(END)
    assert swap.present_value(discount, projection) == pytest.approx(
        floating - fixed
    )


@pytest.mark.parametrize(
    "quotes",
    [
        [SwapQuote(dt.date(2026, 1, 1), 0.03)],
        [SwapQuote(dt.date(2025, 1, 1), 0.02)] * 2,
        [SwapQuote(dt.date(2025, 2, 1), 0.02)],
        [SwapQuote(dt.date(2025, 1, 1), 0.02, 5)],
    ],
)
def test_unsupported_quote_schedules_rejected(quotes: list[SwapQuote]) -> None:
    discount = ConstantShortRateCurve("ois", ORIGIN, 0.01)
    with pytest.raises(ValueError):
        bootstrap_forward_curve(discount, ORIGIN, quotes)


def test_swap_rejects_notional_length() -> None:
    swap = InterestRateSwap(
        "payer",
        0.02,
        100,
        ORIGIN,
        dt.date(2026, 1, 1),
        "12M",
        notional_schedule=[100],
    )
    with pytest.raises(ValueError, match="match payment"):
        swap.payment_schedule()


@pytest.mark.parametrize("convention", list(DayCount))
def test_month_end_periods_reprice_under_named_day_counts(
    convention: DayCount,
) -> None:
    start = dt.date(2024, 1, 1)
    boundary = dt.date(2024, 1, 31)
    end = dt.date(2024, 2, 1)
    curve = ForwardCurve(
        [start, boundary], [boundary, end], np.array([.04, .08]), convention
    )
    assert curve.forward_rate(start, boundary) == pytest.approx(.04)
    assert curve.forward_rate(boundary, end) == pytest.approx(.08)
