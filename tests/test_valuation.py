import datetime as dt
import math

import numpy as np

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.monte_carlo import MonteCarloPricer
from dxquant.payoffs import EuropeanCallPayoff, UpAndOutBarrierCallPayoff
from dxquant.processes import (
    GeometricBrownianMotion,
    MultiAssetGeometricBrownianMotion,
)
from dxquant.valuation import (
    BarrierOptionValuation,
    EuropeanOptionValuation,
    LongstaffSchwartzPricer,
    MultiAssetEuropeanValuation,
)


def test_european_option_valuation_matches_black_scholes() -> None:
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    curve = ConstantShortRateCurve("disc", pricing_date, 0.02)
    env = MarketEnvironment(name="gbm", pricing_date=pricing_date)
    env.add_curve("discount", curve)
    process = GeometricBrownianMotion(drift=0.02, volatility=0.2, seed=123)
    pricer = MonteCarloPricer(
        env, process, discount_curve_key="discount", paths=40000, steps=96
    )
    payoff = EuropeanCallPayoff(strike=105.0)
    valuation = EuropeanOptionValuation(pricer, payoff, maturity, spot=100.0)

    mc_price = valuation.price_mc()
    analytical = valuation.analytical_price()
    assert analytical is not None
    np.testing.assert_allclose(mc_price, analytical, rtol=2e-2)


def test_american_put_longstaff_schwartz() -> None:
    rate = 0.06
    maturity_years = 1.0
    process = GeometricBrownianMotion(drift=rate, volatility=0.2, seed=123)
    curve = ConstantShortRateCurve("disc", dt.date(2024, 1, 1), rate)
    lsm = LongstaffSchwartzPricer(
        process=process, discount_curve=curve, paths=40000, steps=50
    )
    price = lsm.price_american_put(
        spot=40.0, strike=40.0, maturity_years=maturity_years
    )
    np.testing.assert_allclose(price, 2.3, rtol=5e-2)


def test_barrier_option_valuation_matches_vanilla_when_far_barrier() -> None:
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    curve = ConstantShortRateCurve("disc", pricing_date, 0.02)
    env = MarketEnvironment(name="barrier", pricing_date=pricing_date)
    env.add_curve("discount", curve)
    process = GeometricBrownianMotion(drift=0.02, volatility=0.2, seed=321)
    pricer = MonteCarloPricer(
        env, process, discount_curve_key="discount", paths=20000, steps=64
    )
    vanilla = EuropeanOptionValuation(
        pricer, EuropeanCallPayoff(strike=100.0), maturity, spot=100.0
    ).price_mc()
    barrier_payoff = UpAndOutBarrierCallPayoff(strike=100.0, barrier=500.0)
    barrier_price = BarrierOptionValuation(
        pricer, barrier_payoff, maturity, spot=100.0
    ).price()
    np.testing.assert_allclose(barrier_price, vanilla, rtol=3e-2)


def test_multi_asset_valuation_matches_deterministic_payoff() -> None:
    drifts = [0.0, 0.0]
    vols = [0.0, 0.0]
    corr = np.eye(2)
    process = MultiAssetGeometricBrownianMotion(drifts, vols, corr)
    curve = ConstantShortRateCurve("disc", dt.date(2024, 1, 1), 0.01)

    def payoff(terminal: np.ndarray) -> np.ndarray:
        basket = 0.5 * (terminal[:, 0] + terminal[:, 1])
        return np.asarray(np.maximum(basket - 90.0, 0.0), dtype=float)

    valuation = MultiAssetEuropeanValuation(
        process=process,
        discount_curve=curve,
        payoff=payoff,
        maturity_years=1.0,
        spot=[100.0, 80.0],
        paths=5000,
        steps=10,
    )
    price = valuation.price()
    expected = math.exp(-0.01 * 1.0) * max(0.5 * (100 + 80) - 90, 0)
    np.testing.assert_allclose(price, expected, rtol=1e-3)
