import datetime as dt

import numpy as np

from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.monte_carlo import MonteCarloPricer, MonteCarloResult
from dxquant.payoffs import ArithmeticAsianCallPayoff, EuropeanCallPayoff
from dxquant.portfolio import (
    PortfolioPricer,
    PositionSpec,
    Scenario,
    ScenarioAnalyzer,
    scale_spot,
)
from dxquant.processes import GeometricBrownianMotion


def build_zero_vol_pricer(
    rate: float=0.0,
) -> tuple[MonteCarloPricer, dt.date, dt.date]:
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    curve = ConstantShortRateCurve("disc", pricing_date, rate)
    env = MarketEnvironment(name="global", pricing_date=pricing_date)
    env.add_curve("discount", curve)
    process = GeometricBrownianMotion(drift=rate, volatility=0.0)
    pricer = MonteCarloPricer(env, process, "discount", paths=8, steps=8)
    return pricer, pricing_date, maturity


def test_portfolio_pricer_deterministic() -> None:
    pricer, _, maturity = build_zero_vol_pricer()
    portfolio_pricer = PortfolioPricer()
    specs = [
        PositionSpec(
            position_id="long_call",
            pricer=pricer,
            payoff=EuropeanCallPayoff(strike=90.0),
            maturity=maturity,
            spot=100.0,
            quantity=1.0,
        ),
        PositionSpec(
            position_id="short_call",
            pricer=pricer,
            payoff=EuropeanCallPayoff(strike=110.0),
            maturity=maturity,
            spot=100.0,
            quantity=-0.5,
        ),
        PositionSpec(
            position_id="asian",
            pricer=pricer,
            payoff=ArithmeticAsianCallPayoff(strike=95.0),
            maturity=maturity,
            spot=100.0,
            quantity=0.2,
            path_dependent=True,
        ),
    ]

    result = portfolio_pricer.price(specs)
    expected_price = (100.0 - 90.0) + (-0.5 * 0.0) + 0.2 * (100.0 - 95.0)
    np.testing.assert_allclose(result.price, expected_price, rtol=1e-12)
    assert result.value_at_risk(0.95) == 0.0
    assert result.expected_shortfall(0.95) == 0.0


def test_portfolio_var_positive_with_vol() -> None:
    rate = 0.01
    vol = 0.2
    pricing_date = dt.date(2024, 1, 1)
    maturity = dt.date(2025, 1, 1)
    curve = ConstantShortRateCurve("disc", pricing_date, rate)
    env = MarketEnvironment(name="stochastic", pricing_date=pricing_date)
    env.add_curve("discount", curve)
    process = GeometricBrownianMotion(drift=rate, volatility=vol, seed=123)
    pricer = MonteCarloPricer(env, process, "discount", paths=20000, steps=64)
    specs = [
        PositionSpec(
            position_id="call",
            pricer=pricer,
            payoff=EuropeanCallPayoff(strike=80.0),
            maturity=maturity,
            spot=100.0,
            quantity=1.0,
        )
    ]
    portfolio_result = PortfolioPricer().price(specs)
    assert portfolio_result.value_at_risk(0.95) > 0.0
    assert portfolio_result.expected_shortfall(
        0.95
    ) >= portfolio_result.value_at_risk(0.95)


def test_portfolio_multicurrency_conversion() -> None:
    class DummyPricer:
        def __init__(self, value: float):
            self.value = value

        def price_terminal(
            self, *args: object, **kwargs: object
        ) -> MonteCarloResult:
            arr = np.array([self.value])
            return MonteCarloResult(
                price=self.value, std_error=0.0, discounted_payoffs=arr
            )

    usd_spec = PositionSpec(
        position_id="usd",
        pricer=DummyPricer(10.0),
        payoff=EuropeanCallPayoff(strike=0.0),  # unused
        maturity=dt.date(2025, 1, 1),
        spot=0.0,
        quantity=1.0,
        currency="USD",
    )
    eur_spec = PositionSpec(
        position_id="eur",
        pricer=DummyPricer(5.0),
        payoff=EuropeanCallPayoff(strike=0.0),
        maturity=dt.date(2025, 1, 1),
        spot=0.0,
        quantity=1.0,
        currency="EUR",
    )
    result = PortfolioPricer().price(
        [usd_spec, eur_spec], base_currency="USD", fx_rates={"EUR": 1.1}
    )
    np.testing.assert_allclose(result.price, 10.0 + 5.0 * 1.1, rtol=1e-12)
    assert result.std_error == 0.0


def test_scenario_analyzer_applies_shocks() -> None:
    pricer, _, maturity = build_zero_vol_pricer()
    specs = [
        PositionSpec(
            position_id="call",
            pricer=pricer,
            payoff=EuropeanCallPayoff(strike=80.0),
            maturity=maturity,
            spot=100.0,
            quantity=1.0,
        )
    ]
    analyzer = ScenarioAnalyzer(PortfolioPricer(), specs)
    scenarios = [
        Scenario("base", lambda s: s),
        Scenario("down10", scale_spot("call", 0.9)),
    ]
    results = analyzer.run(scenarios)
    assert len(results) == 2
    assert results[0].result.price > results[1].result.price
