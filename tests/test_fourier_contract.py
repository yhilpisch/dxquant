"""Degenerate limits and diagnostics for Carr-Madan quadrature."""

import math

import pytest

from dxquant import (
    FourierPricer,
    GeometricBrownianMotion,
    black_scholes_call_price,
    gbm_characteristic_function,
)


@pytest.mark.parametrize("vol, maturity", [(0, 1), (0.2, 0), (0.2, 1)])
@pytest.mark.parametrize("strike", [80, 100, 120])
def test_fourier_gbm_limits_and_prices(
    vol: float,
    maturity: float,
    strike: float,
) -> None:
    process = GeometricBrownianMotion(0.02, vol)
    cf = gbm_characteristic_function(process, maturity, 100)
    price = FourierPricer().price_call(
        cf, 100, strike, math.exp(-0.02 * maturity)
    )
    expected = black_scholes_call_price(100, strike, 0.02, vol, maturity)
    assert price == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "controls",
    [
        {"dampening": 0},
        {"integration_limit": -1},
        {"steps": 0},
        {"absolute_tolerance": math.nan},
        {"max_tail_steps": 0},
    ],
)
def test_invalid_fourier_controls(controls: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FourierPricer(**controls)  # type: ignore[arg-type]


def test_fourier_rejects_unnormalized_or_missing_moment_cf() -> None:
    with pytest.raises(ValueError, match="normalized CF"):
        FourierPricer().price_call(lambda u: 2 + 0j, 100, 100, 1)
    with pytest.raises(ValueError):
        FourierPricer().price_call(lambda u: complex(math.nan), 100, 100, 1)


def test_insufficient_tail_budget_is_reported() -> None:
    cf = gbm_characteristic_function(GeometricBrownianMotion(0.02, 0.2), 1, 100)
    with pytest.raises(ValueError, match="tail did not converge"):
        FourierPricer(integration_limit=0.01, max_tail_steps=1).price_call(
            cf, 100, 100, math.exp(-0.02)
        )
