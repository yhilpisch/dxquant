"""Joint-scenario, currency, sensitivity, and sampling-error contracts."""

import datetime as dt
import math
from dataclasses import replace

import numpy as np
import pytest

from dxquant import (
    ConstantShortRateCurve,
    EuropeanCallPayoff,
    GeometricBrownianMotion,
    MarketEnvironment,
    MonteCarloPricer,
    PortfolioPricer,
    PositionSpec,
    Scenario,
    ScenarioAnalyzer,
    parallel_greeks,
    parallel_value,
    risk_report,
    scale_spot,
)
from dxquant.valuation import black_scholes_call_price

DATE = dt.date(2024, 1, 1)
MATURITY = dt.date(2025, 1, 1)
T = 366 / 365


def _pricer(
    seed: int | None=7,
    vol: float=0.2,
    antithetic: bool=True,
    matching: bool=True,
) -> MonteCarloPricer:
    env = MarketEnvironment("test", DATE)
    env.add_curve("disc", ConstantShortRateCurve("disc", DATE, 0.02))
    process = GeometricBrownianMotion(
        0.02,
        vol,
        seed=seed,
        antithetic=antithetic,
        moment_matching=matching,
    )
    return MonteCarloPricer(env, process, "disc", paths=2000, steps=1)


def _spec(pricer: MonteCarloPricer) -> PositionSpec:
    return PositionSpec("call", pricer, EuropeanCallPayoff(100), MATURITY, 100)


def test_joint_scenario_identity_required_for_aggregate_risk() -> None:
    spec = _spec(_pricer())
    other = replace(spec, position_id="other", quantity=-1)
    result = PortfolioPricer().price([spec, other])
    assert not result.risk_scenarios_aligned
    with pytest.raises(ValueError, match="joint scenario"):
        result.value_at_risk()
    table = risk_report(PortfolioPricer(), [spec, other])
    assert not table.attrs["joint_scenarios"]
    assert math.isnan(table.iloc[-1]["var_95"])
    declared = [replace(s, scenario_set="joint") for s in (spec, other)]
    result = PortfolioPricer().price(declared)
    assert result.risk_scenarios_aligned
    assert (
        result.price
        == result.value_at_risk()
        == result.expected_shortfall()
        == 0
    )
    later = replace(declared[1], maturity=dt.date(2026, 1, 1))
    assert (
        not PortfolioPricer().price([declared[0], later]).risk_scenarios_aligned
    )


@pytest.mark.parametrize("operation", ["price", "parallel", "greeks"])
def test_duplicate_position_ids_rejected(operation: str) -> None:
    spec = _spec(_pricer())
    with pytest.raises(ValueError, match="unique"):
        if operation == "price":
            PortfolioPricer().price([spec, spec])
        elif operation == "parallel":
            parallel_value([spec, spec], max_workers=1)
        else:
            parallel_greeks([spec, spec], max_workers=1)


def test_parallel_greeks_scale_quantity_and_currency() -> None:
    spec = replace(_spec(_pricer(vol=0)), quantity=-3, currency="EUR")
    fx = {"EUR": 1.2}
    greek = parallel_greeks(
        [spec], max_workers=1, base_currency="USD", fx_rates=fx
    )["call"]
    assert greek == pytest.approx(-3.6)
    table = risk_report(
        PortfolioPricer(), [spec], base_currency="USD", fx_rates=fx
    )
    assert table.iloc[0]["delta"] == pytest.approx(greek)
    assert table.iloc[0]["currency"] == "USD"
    assert table.iloc[0]["position_currency"] == "EUR"


def test_scenario_pnl_is_change_from_base() -> None:
    spec = _spec(_pricer(vol=0))
    results = ScenarioAnalyzer(PortfolioPricer(), [spec]).run(
        [Scenario("base", lambda s: s), Scenario("up", scale_spot("call", 1.1))]
    )
    assert results[0].pnl() == 0
    assert results[1].pnl() == pytest.approx(10)


@pytest.mark.parametrize("antithetic", [False, True])
def test_standard_errors_against_independent_replications(
    antithetic: bool,
) -> None:
    prices, errors = [], []
    target = black_scholes_call_price(100, 100, 0.02, 0.2, T)
    for seed in range(80):
        result = _pricer(
            seed, antithetic=antithetic, matching=False
        ).price_terminal(EuropeanCallPayoff(100), MATURITY, 100)
        prices.append(result.price)
        errors.append(result.std_error)
    prices_arr, errors_arr = np.array(prices), np.array(errors)
    assert 0.75 < errors_arr.mean() / prices_arr.std(ddof=1) < 1.25
    assert np.mean(np.abs(prices_arr - target) <= 1.96 * errors_arr) >= 0.85


def test_moment_matching_does_not_claim_iid_error() -> None:
    pricer = _pricer()
    result = pricer.price_terminal(EuropeanCallPayoff(100), MATURITY, 100)
    assert result.sampling_scheme == "moment-matched"
    assert math.isnan(result.std_error)
    odd = replace(
        pricer,
        paths=101,
        process=replace(pricer.process, moment_matching=False),
    )
    result = odd.price_terminal(EuropeanCallPayoff(100), MATURITY, 100)
    assert math.isnan(result.std_error)


def test_unseeded_sensitivities_replay_draws_without_mutation() -> None:
    pricer = replace(_pricer(seed=None), paths=30000)
    payoff = EuropeanCallPayoff(100)
    delta = pricer.sensitivity(payoff, MATURITY, 100, bump=0.02)
    gamma = pricer.sensitivity(payoff, MATURITY, 100, bump=0.02, order=2)
    d1 = (0.02 + 0.5 * 0.2**2) * T / (0.2 * math.sqrt(T))
    expected_delta = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    expected_gamma = math.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi)
    expected_gamma /= 100 * 0.2 * math.sqrt(T)
    assert abs(delta - expected_delta) < 0.02
    assert abs(gamma - expected_gamma) < 0.004
    assert pricer.process.seed is None


def test_parallel_greeks_process_pool_matches_serial_units() -> None:
    spec = replace(_spec(_pricer(vol=0)), quantity=2, currency="EUR")
    other = replace(spec, position_id="other", quantity=-1)
    serial = parallel_greeks(
        [spec, other], max_workers=1, base_currency="USD", fx_rates={"EUR": 1.2}
    )
    pooled = parallel_greeks(
        [spec, other], max_workers=2, base_currency="USD", fx_rates={"EUR": 1.2}
    )
    assert pooled == pytest.approx(serial)


def test_zero_quantity_has_known_zero_error() -> None:
    spec = replace(_spec(_pricer()), quantity=0)
    result = PortfolioPricer().price([spec])
    assert result.price == result.std_error == 0
    assert result.position_results[0].std_error == 0


def test_joint_pricing_dates_use_calendar_date_contract() -> None:
    first = _spec(_pricer())
    other_pricer = replace(first.pricer, env=replace(
        first.pricer.env, pricing_date=dt.datetime(2024, 1, 1, 12)
    ))
    other = replace(first, pricer=other_pricer, position_id="other")
    specs = [replace(spec, scenario_set="joint") for spec in (first, other)]
    assert PortfolioPricer().price(specs).risk_scenarios_aligned
