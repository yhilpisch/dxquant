import datetime as dt
import math

import numpy as np

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.monte_carlo import MonteCarloPricer
from dxquant.payoffs import (
    ArithmeticAsianCallPayoff,
    BasketCallPayoff,
    EuropeanCallPayoff,
    UpAndOutBarrierCallPayoff,
)
from dxquant.processes import (
    GeometricBrownianMotion,
    HestonProcess,
    MultiAssetGeometricBrownianMotion,
)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    spot: float, strike: float, rate: float, vol: float, maturity: float
) -> float:
    sigma_sqrt_t = vol * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return spot * norm_cdf(d1) - strike * math.exp(-rate * maturity) * norm_cdf(
        d2
    )


def black_scholes_delta(
    spot: float, strike: float, rate: float, vol: float, maturity: float
) -> float:
    sigma_sqrt_t = vol * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity
    ) / sigma_sqrt_t
    return norm_cdf(d1)


def build_pricer(
    rate: float, vol: float, seed: int=123
) -> tuple[MonteCarloPricer, dt.date, dt.date]:
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    curve = ConstantShortRateCurve("disc", pricing_date, rate)
    env = MarketEnvironment(name="global", pricing_date=pricing_date)
    env.add_curve("discount", curve)
    process = GeometricBrownianMotion(drift=rate, volatility=vol, seed=seed)
    pricer = MonteCarloPricer(
        env=env,
        process=process,
        discount_curve_key="discount",
        paths=40000,
        steps=96,
    )
    return pricer, pricing_date, maturity


def test_monte_carlo_pricer_matches_bs() -> None:
    rate = 0.02
    vol = 0.2
    spot = 100.0
    strike = 105.0
    pricer, _, maturity = build_pricer(rate, vol)
    payoff = EuropeanCallPayoff(strike=strike)

    mc_price = pricer.price_terminal(payoff, maturity=maturity, spot=spot).price
    bs_price = black_scholes_call(spot, strike, rate, vol, maturity=1.0)
    np.testing.assert_allclose(mc_price, bs_price, rtol=2e-2)


def test_monte_carlo_delta_matches_bs() -> None:
    rate = 0.01
    vol = 0.25
    spot = 120.0
    strike = 100.0
    pricer, _, maturity = build_pricer(rate, vol)
    payoff = EuropeanCallPayoff(strike=strike)
    mc_delta = pricer.sensitivity(payoff, maturity, spot, order=1)
    bs_delta = black_scholes_delta(spot, strike, rate, vol, maturity=1.0)
    np.testing.assert_allclose(mc_delta, bs_delta, rtol=5e-2)


def test_asian_payoff_with_zero_volatility() -> None:
    rate = 0.0
    vol = 0.0
    spot = 100.0
    strike = 90.0
    pricer, pricing_date, maturity = build_pricer(rate, vol)
    pricer.paths = 1
    pricer.steps = 12
    payoff = ArithmeticAsianCallPayoff(strike=strike)

    result = pricer.price_path_dependent(payoff, maturity, spot=spot)
    # path constant because volatility zero and drift aligned with rate
    expected_avg = spot
    expected_price = (expected_avg - strike) * math.exp(-rate * 1.0)
    np.testing.assert_allclose(result.price, expected_price, rtol=1e-9)


def test_barrier_knockout() -> None:
    rate = 0.0
    vol = 0.0
    spot = 100.0
    strike = 90.0
    pricer, _, maturity = build_pricer(rate, vol)
    pricer.paths = 1
    pricer.steps = 4

    knocked_out = UpAndOutBarrierCallPayoff(strike=strike, barrier=99.0)
    assert pricer.price_path_dependent(knocked_out, maturity, spot).price == 0.0

    active = UpAndOutBarrierCallPayoff(strike=strike, barrier=150.0)
    price_active = pricer.price_path_dependent(active, maturity, spot).price
    assert price_active == (
        spot - strike
    )  # no discounting, deterministic payoff


def test_single_path_std_error_zero() -> None:
    rate = 0.0
    vol = 0.0
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    env = MarketEnvironment(name="single", pricing_date=pricing_date)
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", pricing_date, rate)
    )
    process = GeometricBrownianMotion(drift=rate, volatility=vol, seed=1)
    pricer = MonteCarloPricer(env, process, "discount", paths=1, steps=2)
    payoff = EuropeanCallPayoff(strike=90.0)

    result = pricer.price_terminal(payoff, maturity, spot=100.0)
    assert result.std_error == 0.0


def test_basket_pricing_with_multi_asset_process() -> None:
    rate = 0.0
    drifts = [0.0, 0.0]
    vols = [0.0, 0.0]
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    process = MultiAssetGeometricBrownianMotion(drifts, vols, corr)
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    env = MarketEnvironment(name="global", pricing_date=pricing_date)
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", pricing_date, rate)
    )
    pricer = MonteCarloPricer(env, process, "discount", paths=1, steps=4)
    payoff = BasketCallPayoff(weights=[0.5, 0.5], strike=90.0)

    result = pricer.price_terminal(payoff, maturity, spot=[100.0, 80.0])
    expected = max(0.5 * (100.0 + 80.0) - 90.0, 0.0)
    assert result.price == expected


def test_monte_carlo_delta_with_zero_vol() -> None:
    rate = 0.01
    vol = 0.0
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2024, 7, 1)
    env = MarketEnvironment(name="delta", pricing_date=pricing_date)
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", pricing_date, rate)
    )
    process = GeometricBrownianMotion(drift=rate, volatility=vol, seed=5)
    pricer = MonteCarloPricer(env, process, "discount", paths=100, steps=8)
    payoff = EuropeanCallPayoff(strike=100.0)

    delta = pricer.sensitivity(payoff, maturity, spot=100.0, order=1, bump=1e-3)
    assert 0.99 < delta < 1.01


def test_heston_pricer_with_path_variance_seed() -> None:
    rate = 0.01
    heston = HestonProcess(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.3, drift=rate, seed=321
    )
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    env = MarketEnvironment(name="heston", pricing_date=pricing_date)
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", pricing_date, rate)
    )
    pricer = MonteCarloPricer(env, heston, "discount", paths=20000, steps=64)
    payoff = EuropeanCallPayoff(strike=110.0)
    price = pricer.price_terminal(payoff, maturity, spot=100.0).price
    assert price > 0.0


def test_heston_delta_gamma_finite_difference() -> None:
    rate = 0.02
    heston = HestonProcess(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.4, drift=rate, seed=99
    )
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    env = MarketEnvironment(name="heston", pricing_date=pricing_date)
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", pricing_date, rate)
    )
    pricer = MonteCarloPricer(env, heston, "discount", paths=50000, steps=96)
    payoff = EuropeanCallPayoff(strike=105.0)
    delta = pricer.sensitivity(payoff, maturity, 100.0, order=1, bump=5e-3)
    gamma = pricer.sensitivity(payoff, maturity, 100.0, order=2, bump=5e-3)
    assert abs(delta) < 1.0
    assert gamma > -1.0
