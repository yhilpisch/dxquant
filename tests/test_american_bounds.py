"""American put bounds and an independent binomial benchmark."""

import datetime as dt
import math

import numpy as np
import pytest

from dxquant import ConstantShortRateCurve, GeometricBrownianMotion
from dxquant.valuation import LongstaffSchwartzPricer


def _binomial_put(spot: float, strike: float) -> float:
    """Cox-Ross-Rubinstein American put at r=5%, vol=20%, T=1."""

    steps = 1500
    dt_years = 1 / steps
    up = math.exp(0.2 * math.sqrt(dt_years))
    probability = (math.exp(0.05 * dt_years) - 1 / up) / (up - 1 / up)
    discount = math.exp(-0.05 * dt_years)
    values = spot * up ** (2 * np.arange(steps + 1) - steps)
    option = np.maximum(strike - values, 0)
    for step in range(steps - 1, -1, -1):
        continuation = discount * (
            probability * option[1:] + (1 - probability) * option[:-1]
        )
        values = spot * up ** (2 * np.arange(step + 1) - step)
        option = np.maximum(strike - values, continuation)
    return float(option[0])


def _pricer(rate: float, vol: float) -> LongstaffSchwartzPricer:
    process = GeometricBrownianMotion(rate, vol, seed=123)
    curve = ConstantShortRateCurve("disc", dt.date(2024, 1, 1), rate)
    return LongstaffSchwartzPricer(process, curve, paths=60000, steps=80)


def test_american_immediate_exercise_deterministic_regression() -> None:
    pricer = _pricer(0.1, 0)
    pricer.paths, pricer.steps = 100, 2
    assert pricer.price_american_put(1, 100, 1) == 99


@pytest.mark.parametrize("spot", [70.0, 100.0, 120.0])
def test_american_put_matches_binomial_and_bounds(spot: float) -> None:
    price = _pricer(0.05, 0.2).price_american_put(spot, 100, 1)
    assert max(100 - spot, 0) <= price <= 100
    assert abs(price - _binomial_put(spot, 100)) < 0.15


@pytest.mark.parametrize("spot", [70.0, 100.0, 120.0])
def test_american_expiry_is_intrinsic(spot: float) -> None:
    assert _pricer(0.05, 0.2).price_american_put(spot, 100, 0) == (
        max(100 - spot, 0)
    )


@pytest.mark.parametrize(
    "values", [(-1, 100, 1), (100, 0, 1), (100, 100, -1), (100, 100, math.nan)]
)
def test_american_invalid_domain(values: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        _pricer(0.05, 0.2).price_american_put(*values)
