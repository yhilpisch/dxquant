import datetime as dt
import math

import numpy as np

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.fourier import FourierPricer, gbm_characteristic_function
from dxquant.monte_carlo import MonteCarloPricer
from dxquant.payoffs import EuropeanCallPayoff
from dxquant.processes import GeometricBrownianMotion, HestonProcess


def black_scholes_call(
    spot: float, strike: float, rate: float, vol: float, maturity: float
) -> float:
    sigma_sqrt_t = vol * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    return spot * norm_cdf(d1) - strike * math.exp(-rate * maturity) * norm_cdf(
        d2
    )


def test_fourier_vs_monte_carlo_gbm() -> None:
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    rate = 0.02
    vol = 0.2
    spot = 100.0
    strike = 105.0

    curve = ConstantShortRateCurve("disc", pricing_date, rate)
    env = MarketEnvironment(name="global", pricing_date=pricing_date)
    env.add_curve("discount", curve)

    process = GeometricBrownianMotion(drift=rate, volatility=vol, seed=123)
    payoff = EuropeanCallPayoff(strike=strike)

    from dxquant.monte_carlo import MonteCarloPricer

    pricer = MonteCarloPricer(env, process, "discount", paths=40000, steps=96)
    mc_price = pricer.price_terminal(payoff, maturity, spot).price

    phi = gbm_characteristic_function(process, maturity=1.0, spot=spot)
    discount = curve.discount_factor(maturity)
    fourier_price = FourierPricer().price_call(
        phi, spot=spot, strike=strike, discount_factor=discount
    )

    bs_price = black_scholes_call(spot, strike, rate, vol, 1.0)
    np.testing.assert_allclose(fourier_price, bs_price, rtol=5e-3)
    np.testing.assert_allclose(mc_price, bs_price, rtol=2e-2)


def test_fourier_heston_consistency() -> None:
    spot = 100.0
    strike = 110.0
    rate = 0.01
    maturity = 1.0
    heston = HestonProcess(
        kappa=2.0, theta=0.04, sigma=0.6, rho=-0.5, drift=rate
    )
    phi = heston.characteristic_function(tau=maturity, spot=spot, variance=0.04)

    discount = math.exp(-rate * maturity)
    fourier_price = FourierPricer(
        dampening=1.5, integration_limit=200.0
    ).price_call(phi, spot=spot, strike=strike, discount_factor=discount)

    # Monte Carlo benchmark with many paths and fine time grid. The Heston
    # Euler scheme carries a small discretization bias at high vol-of-vol, so
    # we use a generous but meaningful tolerance (the previous 50% mask hid the
    # characteristic-function bug; ~3-5% is the real Euler/MC error here).
    env = MarketEnvironment(name="heston", pricing_date=dt.date(2024, 1, 1))
    env.add_curve(
        "discount", ConstantShortRateCurve("disc", env.pricing_date, rate)
    )
    pricer = MonteCarloPricer(env, heston, "discount", paths=100000, steps=256)
    payoff = EuropeanCallPayoff(strike=strike)
    mc_result = pricer.price_terminal(
        payoff, maturity=dt.date(2025, 1, 1), spot=spot
    )

    np.testing.assert_allclose(fourier_price, mc_result.price, rtol=6e-2)
