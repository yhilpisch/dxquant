"""Public date, domain, dimension, and calibration-input boundaries."""

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from dxquant import (
    ConstantShortRateCurve,
    DayCount,
    DepositQuote,
    EuropeanCallPayoff,
    EuropeanOptionValuation,
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
    MarketEnvironment,
    MeanRevertingDiffusion,
    MeanVariancePortfolio,
    MonteCarloPricer,
    MultiAssetGeometricBrownianMotion,
    Spline1D,
    SquareRootJumpDiffusion,
    SwapQuote,
    bootstrap_zero_curve,
    linear_interpolate,
    sn_random_numbers,
    year_fractions,
)

DATE = dt.date(2024, 1, 1)


def _simulate(kind: str, grid: list[float], paths: int) -> None:
    if kind == "heston":
        HestonProcess(1, 0.04, 0.2, 0, 0.02).simulate_paths(
            100, 0.04, grid, paths
        )
    elif kind == "multi":
        MultiAssetGeometricBrownianMotion([0], [0.2], np.eye(1)).simulate_paths(
            [100], grid, paths
        )
    elif kind == "jump":
        JumpDiffusionProcess(0, 0.2, 1, 0, 0.1).simulate_paths(100, grid, paths)
    elif kind == "mean":
        MeanRevertingDiffusion(1, 100, 0.2).simulate_paths(100, grid, paths)
    elif kind == "square":
        SquareRootJumpDiffusion(1, 1, 0.2, 1, 0, 0.1).simulate_paths(
            1, grid, paths
        )
    else:
        GeometricBrownianMotion(0, 0.2).simulate_paths(100, grid, paths)


@pytest.mark.parametrize(
    "kind", ["gbm", "multi", "heston", "jump", "mean", "square"]
)
@pytest.mark.parametrize("grid", [[], [0, math.nan], [0, math.inf], [0, -1]])
def test_invalid_process_grids(kind: str, grid: list[float]) -> None:
    with pytest.raises(ValueError):
        _simulate(kind, grid, 10)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_path_counts(count: int) -> None:
    with pytest.raises(ValueError):
        _simulate("gbm", [0, 1], count)


def test_correlation_and_parameter_domains() -> None:
    with pytest.raises(ValueError):
        MultiAssetGeometricBrownianMotion([0, 0], [0.2, 0.2], np.eye(2) * 2)
    with pytest.raises(ValueError):
        MultiAssetGeometricBrownianMotion([], [], np.empty((0, 0)))
    with pytest.raises(ValueError):
        JumpDiffusionProcess(0, 0.2, math.inf, 0, 0.1)
    with pytest.raises(ValueError):
        GeometricBrownianMotion(math.nan, 0.2)


def test_ordered_curve_dates_and_convention_unification() -> None:
    origin = dt.datetime(2024, 1, 1, 12)
    curve = ConstantShortRateCurve("d", origin, 0.03, DayCount.ACT_360)
    target = dt.date(2024, 7, 1)
    assert curve.discount_factor(target) == pytest.approx(
        math.exp(-0.03 * 182 / 360)
    )
    with pytest.raises(ValueError):
        curve.discount_factor(dt.date(2023, 1, 1))
    np.testing.assert_allclose(
        year_fractions([target, origin], DayCount.ACT_360), [0, 182 / 360]
    )
    with pytest.raises(ValueError):
        ConstantShortRateCurve("d", DATE, 0.03, math.nan)


def test_bootstrap_rejects_past_duplicate_and_off_grid_quotes() -> None:
    deposit = DepositQuote(DATE + dt.timedelta(days=365), 0.02)
    for deposits, swaps in [
        ([DepositQuote(dt.date(2023, 1, 1), 0.02), deposit], []),
        ([deposit, deposit], []),
        ([deposit], [SwapQuote(DATE + dt.timedelta(days=700), 0.03)]),
    ]:
        with pytest.raises(ValueError):
            bootstrap_zero_curve(DATE, deposits, swaps)


def test_mc_expiry_and_mixed_date_analytical_clock() -> None:
    env = MarketEnvironment("test", dt.datetime(2024, 1, 1))
    env.add_curve(
        "d", ConstantShortRateCurve("d", DATE, 0.02, DayCount.ACT_360)
    )
    pricer = MonteCarloPricer(
        env,
        GeometricBrownianMotion(0.02, 0, seed=7),
        "d",
        paths=20,
        steps=1,
        day_count=DayCount.ACT_360,
    )
    payoff = EuropeanCallPayoff(90)
    result = pricer.price_terminal(payoff, DATE, 100)
    assert result.price == 10 and result.std_error == 0
    value = EuropeanOptionValuation(pricer, payoff, dt.date(2025, 1, 1), 100)
    assert value.price_mc() == pytest.approx(value.analytical_price())


@pytest.mark.parametrize(
    "data", [[], [[1]], [[1, 2], [math.nan, 3]], [[1, 2], [math.inf, 3]]]
)
def test_mvo_requires_finite_complete_observations(
    data: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        MeanVariancePortfolio(pd.DataFrame(data))


def test_mvo_weight_shape_and_finiteness() -> None:
    model = MeanVariancePortfolio(pd.DataFrame([[0.01, 0.02], [0.03, 0.01]]))
    for weights in [
        np.array([[0.5, 0.5]]),
        np.array([math.nan, 1]),
        np.array([1, 1]),
    ]:
        with pytest.raises(ValueError):
            model.set_weights(weights)


@pytest.mark.parametrize("nodes", [[], [0], [0, math.nan]])
def test_interpolation_requires_valid_nodes(nodes: list[float]) -> None:
    with pytest.raises(ValueError):
        Spline1D(nodes, nodes)
    if len(nodes) != 1:
        with pytest.raises(ValueError):
            linear_interpolate(nodes, nodes, [0])


def test_singleton_normal_draw_is_not_collapsed_to_zero() -> None:
    draws = sn_random_numbers((1, 1), seed=9)
    assert draws.shape == (1, 1)
    assert draws[0, 0] != 0


def test_calibration_rejects_broadcast_target_shape() -> None:
    from dxquant import calibrate_least_squares

    def model(parameters: np.ndarray) -> np.ndarray:
        return np.array([parameters[0], parameters[0] * 2])

    with pytest.raises(ValueError, match="match finite target"):
        calibrate_least_squares(model, [1], [1])
