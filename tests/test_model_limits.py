"""Model limits, independent analytic fixtures, and diffusion benchmarks.

QuantLib 1.44 fixtures below were computed locally on 2026-10-27 using
sabrVolatility, CoxIngersollRoss.discountBond, and AnalyticHestonEngine.
The formulas are independently cross-checked without a QuantLib dependency.
"""

import math

import numpy as np
import pytest

from dxquant import FourierPricer, GeometricBrownianMotion, HestonProcess
from dxquant.processes import SABRProcess, sabr_implied_volatility
from dxquant.rates.cir import CIRShortRateProcess, cir_zero_coupon


@pytest.mark.parametrize("sigma", [0.0, 1e-8, 1e-4])
def test_heston_deterministic_variance_cf_limit(sigma: float) -> None:
    process = HestonProcess(2, 0.04, sigma, -0.4, 0.02)
    phi = process.characteristic_function(1, 100, 0.09)
    integrated = 0.04 + (0.09 - 0.04) * (1 - math.exp(-2)) / 2
    for u in [0, 0.3, 1.5, -1j]:
        expected = np.exp(
            1j * u * (math.log(100) + 0.02) - 0.5 * (u**2 + 1j * u) * integrated
        )
        assert phi(u) == pytest.approx(expected, rel=1e-4, abs=1e-7)


@pytest.mark.parametrize(
    "strike, expected",
    [
        (80, 22.96838736625497),
        (100, 8.406758480047365),
        (120, 1.7366403664483243),
    ],
)
def test_heston_fourier_quantlib_fixture(
    strike: float, expected: float
) -> None:
    process = HestonProcess(2, 0.04, 0.5, -0.4, 0.02)
    phi = process.characteristic_function(1, 100, 0.04)
    price = FourierPricer(integration_limit=250).price_call(
        phi, 100, strike, math.exp(-0.02)
    )
    assert price == pytest.approx(expected, abs=1e-7)


def test_heston_zero_variance_and_expiry() -> None:
    process = HestonProcess(2, 0, 0, -0.4, 0.02, seed=5)
    spots, variances = process.simulate_paths(100, 0, [0, 0.5, 1], 100)
    np.testing.assert_allclose(spots[:, -1], 100 * math.exp(0.02))
    assert np.all(variances == 0)
    phi = process.characteristic_function(0, 100, 0.04)
    assert phi(0.3) == pytest.approx(np.exp(0.3j * math.log(100)))


def test_heston_zero_sigma_paths_match_constant_variance_gbm() -> None:
    heston = HestonProcess(2, 0.04, 0, -0.4, 0.02, seed=7)
    paths, variance = heston.simulate_paths(100, 0.04, [0, 1], 40000)
    logs = np.log(paths[:, -1] / 100)
    assert abs(logs.mean()) < 1e-12
    assert abs(logs.var() - 0.04) < 1e-12
    np.testing.assert_allclose(variance, 0.04)


@pytest.mark.parametrize("sigma", [0.0, 1e-10, 1e-5])
def test_cir_deterministic_limit(sigma: float) -> None:
    integral = 0.05 * 2 + (0.03 - 0.05) * (1 - math.exp(-1.6)) / 0.8
    assert cir_zero_coupon(0.03, 0.8, 0.05, sigma, 2) == pytest.approx(
        math.exp(-integral), abs=1e-10
    )


@pytest.mark.parametrize(
    "maturity, expected",
    [
        (0.5, 0.9833926533499605),
        (2, 0.9234329239317807),
        (10, 0.6259478266002177),
    ],
)
def test_cir_quantlib_fixture(maturity: float, expected: float) -> None:
    assert cir_zero_coupon(0.03, 0.8, 0.05, 0.15, maturity) == (
        pytest.approx(expected, abs=1e-13)
    )


def test_cir_expiry_long_horizon_and_zero_reversion() -> None:
    assert cir_zero_coupon(0.03, 0.8, 0.05, 0.15, 0) == 1
    assert 0 <= cir_zero_coupon(0.03, 0.8, 0.05, 0.15, 10000) <= 1
    assert cir_zero_coupon(0.03, 0, 0.05, 0, 2) == math.exp(-0.06)
    paths = CIRShortRateProcess(0.8, 0.05, 0, seed=1).simulate_paths(
        0.03, 2, 4, 10
    )
    expected = 0.05 + (0.03 - 0.05) * np.exp(-0.8 * np.linspace(0, 2, 5))
    np.testing.assert_allclose(paths, np.broadcast_to(expected, paths.shape))


def test_sabr_constant_volatility_limits() -> None:
    # Beta=1, nu=0 is GBM; identical seeds give identical diffusion paths.
    sabr = SABRProcess(100, 0.2, 1, -0.3, 0, 0.02, seed=11)
    paths = sabr.simulate_paths(1, 1, 10000)
    gbm = GeometricBrownianMotion(0.02, 0.2, seed=11)
    # In one step both generators use the same first-factor draws.
    logs = np.log(paths[:, -1] / 100)
    assert abs(logs.mean()) < 1e-12
    assert abs(logs.var() - 0.04) < 1e-12
    np.testing.assert_allclose(paths, gbm.simulate_paths(100, [0, 1], 10000))
    normal = SABRProcess(0, 2, 0, 1, 0, 0, seed=13)
    values = normal.simulate_paths(1, 1, 10000)[:, -1]
    assert abs(values.mean()) < 1e-12
    assert abs(values.var() - 4) < 1e-12
    assert np.any(values < 0)


def test_sabr_volatility_diffusion_integrated_second_moment() -> None:
    process = SABRProcess(100, 2, 0, 0.3, 0.8, 0, seed=17)
    terminal = process.simulate_paths(1, 128, 40000)[:, -1]
    expected = 4 * math.expm1(0.8**2) / 0.8**2
    assert abs(terminal.mean() - 100) < 6 * math.sqrt(expected / 40000)
    assert terminal.var() == pytest.approx(expected, rel=0.08)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_sabr_zero_nu_and_near_atm(strike: float) -> None:
    assert sabr_implied_volatility(100, strike, 1, 0.2, 1, -0.4, 0) == (
        pytest.approx(0.2)
    )
    center = sabr_implied_volatility(100, 100, 1, 0.2, 0.5, -0.4, 0.3)
    nearby = sabr_implied_volatility(
        100, 100 * (1 + 1e-10), 1, 0.2, 0.5, -0.4, 0.3
    )
    assert nearby == pytest.approx(center, rel=1e-8)


@pytest.mark.parametrize(
    "beta, alpha, expected",
    [
        (0, 0.005, [0.4050192458871999, 0.20528, 0.17664549778511798]),
        (
            0.5,
            0.04,
            [0.411131893213915, 0.257234978806119, 0.23579327170755532],
        ),
        (
            1,
            0.2,
            [0.3001181632539895, 0.20221333333333336, 0.23357029087986922],
        ),
    ],
)
def test_hagan_quantlib_fixtures(
    beta: float,
    alpha: float,
    expected: list[float],
) -> None:
    result = [
        sabr_implied_volatility(0.025, strike, 2, alpha, beta, -0.3, 0.4)
        for strike in (0.01, 0.025, 0.06)
    ]
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize("rho", [-0.8, 0.8])
def test_heston_discounted_mean_is_martingale(rho: float) -> None:
    process = HestonProcess(
        2,
        0.04,
        0.5,
        rho,
        0.02,
        seed=19,
        antithetic=False,
        moment_matching=False,
    )
    spots, _ = process.simulate_paths(100, 0.04, np.linspace(0, 1, 65), 30000)
    discounted = math.exp(-0.02) * spots[:, -1]
    error = 6 * discounted.std(ddof=1) / math.sqrt(discounted.size)
    assert abs(discounted.mean() - 100) < error
