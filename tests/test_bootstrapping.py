import datetime as dt

import numpy as np

from dxquant.bootstrapping import DepositQuote, SwapQuote, bootstrap_zero_curve


def test_bootstrap_deposits_only_matches_constant_rate() -> None:
    pricing_date = dt.date(2024, 1, 1)
    deposits = [
        DepositQuote(pricing_date + dt.timedelta(days=182), 0.02),
        DepositQuote(pricing_date + dt.timedelta(days=365), 0.025),
    ]
    curve = bootstrap_zero_curve(pricing_date, deposits, [])
    df = curve.discount_factor(pricing_date + dt.timedelta(days=365))
    np.testing.assert_allclose(df, np.exp(-0.025 * 1.0), rtol=1e-6)


def test_bootstrap_with_swap_extends_curve() -> None:
    pricing_date = dt.date(2024, 1, 1)
    deposits = [DepositQuote(pricing_date + dt.timedelta(days=365), 0.02)]
    swaps = [
        SwapQuote(
            pricing_date + dt.timedelta(days=730),
            rate=0.03,
            payments_per_year=1,
        )
    ]
    curve = bootstrap_zero_curve(pricing_date, deposits, swaps)
    df = curve.discount_factor(pricing_date + dt.timedelta(days=730))
    # Manual two-year DF from simple swap bootstrapping
    tau = 1.0
    df1 = np.exp(-0.02 * 1.0)
    expected_df2 = (1.0 - 0.03 * tau * df1) / (1.0 + 0.03 * tau)
    np.testing.assert_allclose(df, expected_df2, rtol=1e-6)


def test_semiannual_swap_reprices_at_par() -> None:
    pricing_date = dt.date(2024, 1, 1)
    first_coupon = pricing_date + dt.timedelta(days=180)
    maturity = pricing_date + dt.timedelta(days=360)
    curve = bootstrap_zero_curve(
        pricing_date,
        [DepositQuote(first_coupon, 0.02)],
        [SwapQuote(maturity, 0.03, payments_per_year=2)],
        day_count=360.0,
    )
    first_df = curve.discount_factor(first_coupon)
    final_df = curve.discount_factor(maturity)
    fixed_pv = 0.03 * 0.5 * (first_df + final_df)
    floating_pv = 1.0 - final_df
    np.testing.assert_allclose(fixed_pv, floating_pv, atol=1e-12)
