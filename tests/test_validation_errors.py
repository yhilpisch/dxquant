# Validation and error-path coverage for edge cases.

import datetime as dt

import numpy as np
import pytest

from dxquant.bootstrapping import DepositQuote, SwapQuote, bootstrap_zero_curve
from dxquant.calendar import BusinessDayCalendar
from dxquant.curves import ConstantShortRateCurve
from dxquant.env import MarketEnvironment
from dxquant.interpolate import Spline1D, linear_interpolate
from dxquant.monte_carlo import MonteCarloPricer, MonteCarloResult
from dxquant.payoffs import EuropeanCallPayoff, UpAndOutBarrierCallPayoff
from dxquant.portfolio import (
    PortfolioPricer,
    PositionSpec,
    Scenario,
    ScenarioAnalyzer,
    scale_spot,
)
from dxquant.processes import (
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
)
from dxquant.random import sn_random_numbers
from dxquant.rates.forward import ForwardCurve
from dxquant.time import (
    ensure_datetime_array,
    uniform_time_grid,
    year_fractions,
)


@pytest.mark.parametrize("path_dependent", [False, True])
def test_mc_rejects_past_maturity(path_dependent: bool) -> None:
    pricing_date = dt.date(2024, 1, 1)
    env = MarketEnvironment("past", pricing_date)
    process = GeometricBrownianMotion(0.02, 0.2, seed=1)
    pricer = MonteCarloPricer(env, process, "discount", paths=100)
    with pytest.raises(ValueError, match="end must not precede start"):
        if path_dependent:
            pricer.price_path_dependent(
                UpAndOutBarrierCallPayoff(100, 120),
                maturity=dt.date(2023, 1, 1),
                spot=100,
            )
        else:
            pricer.price_terminal(
                EuropeanCallPayoff(100),
                maturity=dt.date(2023, 1, 1),
                spot=100,
            )


@pytest.mark.parametrize("lengths", [(1, 1, 2), (1, 2, 2), (2, 1, 2)])
def test_forward_curve_rejects_misaligned_vectors(
    lengths: tuple[int, int, int],
) -> None:
    date = dt.date(2024, 1, 1)
    starts, ends, rates = lengths
    with pytest.raises(ValueError, match="must align"):
        ForwardCurve([date] * starts, [date] * ends, np.ones(rates), 365.0)


def test_sn_random_numbers_shape_guards() -> None:
    with pytest.raises(ValueError):
        sn_random_numbers(())
    with pytest.raises(ValueError):
        sn_random_numbers((10,), antithetic=True)


def test_time_helpers_error_paths() -> None:
    with pytest.raises(TypeError):
        ensure_datetime_array([123])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        ensure_datetime_array([])
    with pytest.raises(ValueError):
        year_fractions([dt.date(2024, 1, 1)], day_count=0)
    with pytest.raises(ValueError):
        uniform_time_grid(1.0, 0)


def test_interpolation_validation_errors() -> None:
    with pytest.raises(ValueError):
        Spline1D([0.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        linear_interpolate([0.0, 1.0], [1.0], [0.5])


def test_bootstrap_validation_errors() -> None:
    pricing_date = dt.date(2024, 1, 1)
    with pytest.raises(ValueError):
        bootstrap_zero_curve(pricing_date, [], [])
    bad_deposit = [DepositQuote(maturity=pricing_date, rate=0.01)]
    with pytest.raises(ValueError):
        bootstrap_zero_curve(pricing_date, bad_deposit, [])
    swap_only = [
        SwapQuote(
            maturity=pricing_date + dt.timedelta(days=365),
            rate=0.02,
            payments_per_year=2,
        )
    ]
    with pytest.raises(ValueError):
        bootstrap_zero_curve(pricing_date, [], swap_only)


def test_calendar_holiday_conversion_and_business_day() -> None:
    holiday = dt.datetime(2024, 1, 1, 15, 0)
    cal = BusinessDayCalendar(name="test", holidays=[holiday])
    assert not cal.is_business_day(dt.date(2024, 1, 1))


def test_process_validation_paths() -> None:
    gbm = GeometricBrownianMotion(drift=0.0, volatility=0.1)
    with pytest.raises(ValueError):
        gbm.simulate_paths(spot=100.0, time_grid=[0.1, 0.0], paths=1)
    with pytest.raises(ValueError):
        gbm.simulate_paths(spot=-1.0, time_grid=[0.0, 0.1], paths=1)
    with pytest.raises(ValueError):
        GeometricBrownianMotion(drift=0.0, volatility=-0.1).simulate_paths(
            spot=100.0, time_grid=[0.0, 0.1], paths=1
        )

    with pytest.raises(ValueError):
        HestonProcess(kappa=0.0, theta=0.04, sigma=0.2, rho=0.0, drift=0.0)
    heston = HestonProcess(kappa=1.0, theta=0.04, sigma=0.2, rho=0.0, drift=0.0)
    with pytest.raises(ValueError):
        heston.simulate_paths(
            spot=0.0, variance=0.04, time_grid=[0.0, 0.1], paths=1
        )
    with pytest.raises(ValueError):
        heston.simulate_paths(
            spot=100.0, variance=-0.01, time_grid=[0.0, 0.1], paths=1
        )

    with pytest.raises(ValueError):
        JumpDiffusionProcess(
            drift=0.0,
            volatility=-0.1,
            jump_intensity=0.5,
            jump_mean=0.0,
            jump_std=0.1,
        )
    jd = JumpDiffusionProcess(
        drift=0.0,
        volatility=0.1,
        jump_intensity=0.5,
        jump_mean=0.0,
        jump_std=0.1,
    )
    with pytest.raises(ValueError):
        jd.simulate_paths(spot=100.0, time_grid=[0.5, 0.5], paths=1)


def test_monte_carlo_type_errors_and_sensitivity_guard() -> None:
    pricing_date = dt.date(2024, 1, 1)
    env = MarketEnvironment(name="mc", pricing_date=pricing_date)
    env.add_curve("disc", ConstantShortRateCurve("disc", pricing_date, 0.0))

    heston = HestonProcess(kappa=1.0, theta=0.04, sigma=0.2, rho=0.0, drift=0.0)
    maturity = pricing_date + dt.timedelta(days=30)
    pricer = MonteCarloPricer(env, heston, "disc", paths=10, steps=4)
    with pytest.raises(TypeError):
        pricer.price_terminal(EuropeanCallPayoff(100.0), maturity, spot=[100.0])

    gbm = GeometricBrownianMotion(drift=0.0, volatility=0.1)
    pricer_gbm = MonteCarloPricer(env, gbm, "disc", paths=10, steps=4)
    with pytest.raises(TypeError):
        pricer_gbm.price_terminal(
            EuropeanCallPayoff(100.0), maturity, spot=[100.0]
        )
    with pytest.raises(ValueError):
        pricer_gbm.sensitivity(
            EuropeanCallPayoff(100.0), maturity, 100.0, order=3
        )

    # price_european helper should still work for scalar processes
    euro_price = pricer_gbm.price_european(
        EuropeanCallPayoff(100.0), maturity, spot=100.0
    )
    assert euro_price >= 0.0


def test_portfolio_error_paths_and_scenarios() -> None:
    pricer = PortfolioPricer()
    with pytest.raises(ValueError):
        pricer.price([])

    class DummyPricer:
        def price_terminal(
            self, *args: object, **kwargs: object
        ) -> MonteCarloResult:
            arr = np.array([1.0, 2.0])
            return MonteCarloResult(
                price=1.5, std_error=0.0, discounted_payoffs=arr
            )

    base_spec = PositionSpec(
        position_id="p1",
        pricer=DummyPricer(),
        payoff=EuropeanCallPayoff(0.0),
        maturity=dt.date(2024, 1, 1),
        spot=0.0,
    )
    with pytest.raises(ValueError):
        pricer.price([base_spec], base_currency="USD", fx_rates={"EUR": 1.1})

    scenario = Scenario("scale", scale_spot("p1", 2.0))
    analyzer = ScenarioAnalyzer(pricer, [base_spec])
    results = analyzer.run([scenario])
    assert results[0].pnl() == pytest.approx(0.0)
