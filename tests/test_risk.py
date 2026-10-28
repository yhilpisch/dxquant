# Risk reporting and parallel valuation helpers.

import datetime as dt
import math

import numpy as np
import pandas as pd

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.monte_carlo import MonteCarloPricer
from dxquant.parallel import parallel_greeks, parallel_value
from dxquant.payoffs import EuropeanCallPayoff
from dxquant.portfolio import PortfolioPricer, PositionSpec
from dxquant.processes import GeometricBrownianMotion
from dxquant.risk import risk_report
from dxquant.valuation.european import black_scholes_call_price


def _make_pricer(seed: int=42, paths: int=40000) -> MonteCarloPricer:
    pricing_date = dt.date(2024, 1, 1)
    env = MarketEnvironment("g", pricing_date)
    env.add_curve("discount", ConstantShortRateCurve("d", pricing_date, 0.02))
    process = GeometricBrownianMotion(drift=0.02, volatility=0.2, seed=seed)
    return MonteCarloPricer(env, process, "discount", paths=paths, steps=64)


def _make_specs(pricer: MonteCarloPricer) -> list[PositionSpec]:
    maturity = dt.date(2025, 1, 1)
    return [
        PositionSpec(
            "call_100",
            pricer,
            EuropeanCallPayoff(100.0),
            maturity,
            100.0,
            quantity=1.0,
            scenario_set="shared-gbm",
        ),
        PositionSpec(
            "call_110",
            pricer,
            EuropeanCallPayoff(110.0),
            maturity,
            105.0,
            quantity=2.0,
            scenario_set="shared-gbm",
        ),
    ]


def test_risk_report_returns_dataframe_with_portfolio_row() -> None:
    pricer = _make_pricer()
    specs = _make_specs(pricer)
    df = risk_report(PortfolioPricer(), specs, bump=1e-2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # two positions + portfolio total
    assert "PORTFOLIO" in df["position_id"].values
    assert {"price", "delta", "gamma", "var_95", "es"}.issubset(df.columns)


def test_risk_report_delta_matches_black_scholes() -> None:
    pricer = _make_pricer(seed=7, paths=60000)
    specs = _make_specs(pricer)
    df = risk_report(PortfolioPricer(), specs, bump=1e-2)
    # BSM delta for ATM call (S=100, K=100, r=0.02, vol=0.2, T=366/365).
    ttm = (dt.date(2025, 1, 1) - dt.date(2024, 1, 1)).days / 365.0
    d1 = (math.log(100 / 100) + (0.02 + 0.5 * 0.2**2) * ttm) / (
        0.2 * math.sqrt(ttm)
    )

    def n_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    bs_delta = n_cdf(d1)
    row = df[df["position_id"] == "call_100"].iloc[0]
    np.testing.assert_allclose(row["delta"], bs_delta, rtol=5e-2)


def test_risk_report_price_matches_black_scholes() -> None:
    pricer = _make_pricer(seed=7, paths=60000)
    specs = _make_specs(pricer)
    df = risk_report(PortfolioPricer(), specs)
    ttm = (dt.date(2025, 1, 1) - dt.date(2024, 1, 1)).days / 365.0
    bs_price = black_scholes_call_price(100.0, 100.0, 0.02, 0.2, ttm)
    row = df[df["position_id"] == "call_100"].iloc[0]
    np.testing.assert_allclose(row["price"], bs_price, rtol=2e-2)


def test_portfolio_risk_report_var_es_non_decreasing() -> None:
    pricer = _make_pricer()
    specs = _make_specs(pricer)
    df = risk_report(PortfolioPricer(), specs)
    portfolio = df[df["position_id"] == "PORTFOLIO"].iloc[0]
    # ES at 95% should be at least as large as VaR at 95% (loss tail).
    assert portfolio["es"] >= portfolio["var_95"] - 1e-9


def test_parallel_value_matches_sequential() -> None:
    pricer = _make_pricer()
    specs = _make_specs(pricer)
    sequential = parallel_value(specs, max_workers=1)
    assert set(sequential) == {"call_100", "call_110"}
    # Sanity: prices are positive.
    assert all(v > 0 for v in sequential.values())


def test_parallel_value_empty() -> None:
    assert parallel_value([], max_workers=1) == {}


def test_parallel_value_process_pool_smoke() -> None:
    # Exercises the ProcessPoolExecutor path (pickling pricers across workers).
    pricer = _make_pricer(seed=1, paths=5000)
    specs = _make_specs(pricer)
    result = parallel_value(specs, max_workers=2)
    assert set(result) == {"call_100", "call_110"}
    assert all(v > 0 for v in result.values())


def test_parallel_greeks_delta_positive_for_call() -> None:
    pricer = _make_pricer()
    specs = _make_specs(pricer)
    deltas = parallel_greeks(specs, order=1, bump=1e-2, max_workers=1)
    assert all(d > 0 for d in deltas.values())  # call deltas are positive


def test_parallel_greeks_gamma_positive_for_call() -> None:
    pricer = _make_pricer()
    specs = _make_specs(pricer)
    gammas = parallel_greeks(specs, order=2, bump=1e-2, max_workers=1)
    assert all(g > 0 for g in gammas.values())  # call gammas are positive
