import datetime as dt

import numpy as np

from dxquant.bootstrapping import SwapQuote
from dxquant.curves import InterpolatedZeroCurve
from dxquant.rates.cir import (
    CIRShortRateProcess,
    calibrate_cir,
    cir_zero_coupon,
)
from dxquant.rates.forward import bootstrap_forward_curve
from dxquant.rates.swap import InterestRateSwap
from dxquant.time import DayCount


def test_cir_zero_coupon_matches_known_value() -> None:
    price = cir_zero_coupon(
        short_rate=0.05, kappa=0.6, theta=0.04, sigma=0.1, maturity=2.0
    )
    np.testing.assert_allclose(price, 0.9127, rtol=1e-4)


def test_cir_short_rate_paths_positive() -> None:
    process = CIRShortRateProcess(kappa=0.8, theta=0.05, sigma=0.15, seed=123)
    paths = process.simulate_paths(
        short_rate=0.03, maturity=1.0, steps=50, paths=1000
    )
    assert np.all(paths >= 0.0)


def test_interest_rate_swap_par_value() -> None:
    pricing_date = dt.date(2024, 1, 1)
    nodes = [
        (pricing_date + dt.timedelta(days=365 * 1), 0.025),
        (pricing_date + dt.timedelta(days=365 * 2), 0.025),
        (pricing_date + dt.timedelta(days=365 * 3), 0.025),
    ]
    curve = InterpolatedZeroCurve("zc", pricing_date, nodes)
    swap = InterestRateSwap(
        name="par",
        fixed_rate=0.0,  # placeholder
        notional=1_000_000,
        effective_date=pricing_date,
        termination_date=pricing_date.replace(year=2026),
        payment_frequency="12M",
    )
    par_rate = swap.par_rate(curve)
    swap.fixed_rate = par_rate
    pv = swap.present_value(curve)
    assert abs(pv) < 1e-2


def test_calibrate_cir_recovers_parameters() -> None:
    true_params = np.array([0.8, 0.05, 0.15])
    maturities = np.array([0.5, 1.0, 2.0, 3.0])
    params_list = true_params.tolist()
    targets = [
        cir_zero_coupon(
            short_rate=0.03,
            kappa=params_list[0],
            theta=params_list[1],
            sigma=params_list[2],
            maturity=float(t),
        )
        for t in maturities
    ]
    fitted = calibrate_cir(
        0.03, maturities.tolist(), targets, initial_guess=[0.5, 0.04, 0.1]
    )
    np.testing.assert_allclose(fitted, true_params, rtol=0.2)


def test_swap_with_day_count_enum_matches_float() -> None:
    pricing_date = dt.date(2024, 1, 1)
    nodes = [
        (pricing_date + dt.timedelta(days=365), 0.025),
        (pricing_date + dt.timedelta(days=365 * 2), 0.025),
    ]
    curve = InterpolatedZeroCurve("zc", pricing_date, nodes)
    swap_enum = InterestRateSwap(
        name="s",
        fixed_rate=0.025,
        notional=1_000_000,
        effective_date=pricing_date,
        termination_date=pricing_date.replace(year=2026),
        payment_frequency="12M",
        day_count=DayCount.ACT_365,
    )
    swap_float = InterestRateSwap(
        name="s",
        fixed_rate=0.025,
        notional=1_000_000,
        effective_date=pricing_date,
        termination_date=pricing_date.replace(year=2026),
        payment_frequency="12M",
        day_count=365.0,
    )
    assert (
        abs(swap_enum.present_value(curve) - swap_float.present_value(curve))
        < 1e-9
    )


def test_amortizing_swap_notional_schedule_used() -> None:
    pricing_date = dt.date(2024, 1, 1)
    # Non-flat curve so amortizing vs constant par rates differ.
    nodes = [
        (pricing_date.replace(year=2025), 0.02),
        (pricing_date.replace(year=2026), 0.03),
        (pricing_date.replace(year=2027), 0.04),
    ]
    curve = InterpolatedZeroCurve("zc", pricing_date, nodes)
    swap = InterestRateSwap(
        name="amort",
        fixed_rate=0.03,
        notional=1_000_000,
        effective_date=pricing_date,
        termination_date=pricing_date.replace(year=2027),
        payment_frequency="12M",
        notional_schedule=[1_000_000, 500_000, 250_000],
    )
    par = swap.par_rate(curve)
    swap.fixed_rate = par
    assert abs(swap.present_value(curve)) < 1e-2  # par swap has ~zero PV
    # Amortizing par rate differs from the constant-notional par rate.
    const = InterestRateSwap(
        name="const",
        fixed_rate=0.03,
        notional=1_000_000,
        effective_date=pricing_date,
        termination_date=pricing_date.replace(year=2027),
        payment_frequency="12M",
    )
    assert abs(par - const.par_rate(curve)) > 1e-4


def test_bootstrap_forward_curve_recovers_par_swaps() -> None:
    pricing_date = dt.date(2024, 1, 1)
    nodes = [
        (pricing_date.replace(year=2025), 0.02),
        (pricing_date.replace(year=2026), 0.02),
        (pricing_date.replace(year=2027), 0.02),
    ]
    discount = InterpolatedZeroCurve("ois", pricing_date, nodes)
    swaps = [
        SwapQuote(pricing_date.replace(year=2025), 0.02, 1),
        SwapQuote(pricing_date.replace(year=2026), 0.02, 1),
        SwapQuote(pricing_date.replace(year=2027), 0.02, 1),
    ]
    fwd = bootstrap_forward_curve(discount, pricing_date, swaps)
    assert len(fwd.forward_rates) == 3
    # Simply-compounded forwards from a flat 2% cont curve are ~2% (slightly
    # above).
    np.testing.assert_allclose(fwd.forward_rates, 0.02, rtol=2e-3)
    assert np.all(fwd.forward_rates > 0)


def test_bootstrap_forward_curve_consistent_with_swap_pv() -> None:
    from dxquant.time import year_fraction

    pricing_date = dt.date(2024, 1, 1)
    nodes = [
        (pricing_date.replace(year=2025), 0.02),
        (pricing_date.replace(year=2026), 0.025),
        (pricing_date.replace(year=2027), 0.03),
    ]
    discount = InterpolatedZeroCurve("ois", pricing_date, nodes)
    par_rates = []
    for years in (1, 2, 3):
        s = InterestRateSwap(
            name=f"s{years}",
            fixed_rate=0.0,
            notional=1.0,
            effective_date=pricing_date,
            termination_date=pricing_date.replace(year=2024 + years),
            payment_frequency="12M",
        )
        par_rates.append((years, s.par_rate(discount)))
    swaps = [
        SwapQuote(pricing_date.replace(year=2024 + y), r, 1)
        for y, r in par_rates
    ]
    fwd = bootstrap_forward_curve(discount, pricing_date, swaps)
    # Reconstruct each swap's PV using the bootstrapped forwards (same calendar
    # dates).
    schedule = [pricing_date] + [
        pricing_date.replace(year=2024 + y) for y in (1, 2, 3)
    ]
    for k in (1, 2, 3):
        par = par_rates[k - 1][1]
        fixed_pv = 0.0
        float_pv = 0.0
        for i in range(1, k + 1):
            tau = year_fraction(schedule[i - 1], schedule[i], DayCount.ACT_365)
            df = discount.discount_factor(schedule[i])
            fixed_pv += par * tau * df
            float_pv += fwd.forward_rates[i - 1] * tau * df
        np.testing.assert_allclose(float_pv - fixed_pv, 0.0, atol=1e-9)
