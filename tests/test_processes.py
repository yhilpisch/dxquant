import numpy as np

from dxquant.processes import (
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
    MeanRevertingDiffusion,
    MultiAssetGeometricBrownianMotion,
    SquareRootJumpDiffusion,
    sabr_implied_volatility,
)
from dxquant.time import uniform_time_grid


def test_gbm_expected_mean_matches_theory() -> None:
    process = GeometricBrownianMotion(drift=0.05, volatility=0.2, seed=7)
    grid = uniform_time_grid(1.0, 50)
    paths = process.simulate_paths(spot=100.0, time_grid=grid, paths=20000)
    terminal = paths[:, -1]
    theoretical_mean = 100.0 * np.exp(0.05 * 1.0)
    np.testing.assert_allclose(terminal.mean(), theoretical_mean, rtol=5e-3)


def test_gbm_rejects_invalid_inputs() -> None:
    process = GeometricBrownianMotion(drift=0.0, volatility=0.2)
    grid = [0.0, 0.5, 0.2]
    try:
        process.simulate_paths(spot=100.0, time_grid=grid, paths=1)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unsorted grid")


def test_multi_asset_gbm_correlation_structure() -> None:
    drifts = [0.03, 0.05]
    vols = [0.2, 0.3]
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    process = MultiAssetGeometricBrownianMotion(drifts, vols, corr, seed=123)
    grid = uniform_time_grid(1.0, 20)
    paths = process.simulate_paths(
        spot=[100.0, 80.0], time_grid=grid, paths=20000
    )
    terminal = paths[:, -1, :]
    log_returns = np.log(terminal / np.array([100.0, 80.0]))
    sample_corr = np.corrcoef(log_returns.T)
    np.testing.assert_allclose(sample_corr, corr, rtol=5e-2)


def test_heston_paths_non_negative_variance() -> None:
    process = HestonProcess(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.3, drift=0.01, seed=42
    )
    grid = uniform_time_grid(0.5, 20)
    _, variances = process.simulate_paths(
        spot=100.0, variance=0.04, time_grid=grid, paths=500
    )
    assert (variances >= 0.0).all()


def test_jump_diffusion_paths_positive() -> None:
    process = JumpDiffusionProcess(
        drift=0.02,
        volatility=0.2,
        jump_intensity=0.5,
        jump_mean=-0.1,
        jump_std=0.2,
        seed=123,
    )
    grid = uniform_time_grid(1.0, 50)
    paths = process.simulate_paths(spot=100.0, time_grid=grid, paths=2000)
    assert np.all(paths > 0)


def test_heston_characteristic_function_returns_complex() -> None:
    process = HestonProcess(
        kappa=1.5, theta=0.04, sigma=0.3, rho=-0.2, drift=0.02
    )
    phi = process.characteristic_function(tau=1.0, spot=100.0, variance=0.04)
    value = phi(0.5)
    assert isinstance(value, complex)
    assert value != 0.0


def test_sabr_implied_volatility_positive() -> None:
    vol = sabr_implied_volatility(
        forward=100.0,
        strike=100.0,
        maturity=1.0,
        alpha=0.2,
        beta=0.5,
        rho=0.0,
        vol_vol=0.3,
    )
    assert vol > 0


def test_mean_reverting_diffusion_pulls_toward_theta() -> None:
    process = MeanRevertingDiffusion(
        kappa=5.0, theta=10.0, volatility=0.1, seed=42
    )
    grid = uniform_time_grid(5.0, 200)  # long horizon -> strong mean reversion
    paths = process.simulate_paths(
        initial_value=20.0, time_grid=grid, paths=5000
    )
    # Terminal mean should be close to theta (kappa*T large).
    np.testing.assert_allclose(paths[:, -1].mean(), 10.0, rtol=5e-2)


def test_mean_reverting_diffusion_rejects_invalid_inputs() -> None:
    import pytest

    with pytest.raises(ValueError):
        MeanRevertingDiffusion(kappa=-1.0, theta=1.0, volatility=0.1)
    with pytest.raises(ValueError):
        MeanRevertingDiffusion(kappa=1.0, theta=1.0, volatility=-0.1)


def test_square_root_jump_diffusion_non_negative() -> None:
    process = SquareRootJumpDiffusion(
        kappa=2.0,
        theta=0.04,
        volatility=0.3,
        jump_intensity=0.5,
        jump_mean=-0.1,
        jump_std=0.2,
        seed=7,
    )
    grid = uniform_time_grid(1.0, 50)
    paths = process.simulate_paths(
        initial_value=0.04, time_grid=grid, paths=2000
    )
    assert np.all(paths >= 0.0)


def test_square_root_jump_diffusion_mean_reverts_to_theta() -> None:
    process = SquareRootJumpDiffusion(
        kappa=5.0,
        theta=0.05,
        volatility=0.1,
        jump_intensity=0.0,
        jump_mean=0.0,
        jump_std=0.0,
        seed=11,
    )
    grid = uniform_time_grid(5.0, 200)
    paths = process.simulate_paths(
        initial_value=0.2, time_grid=grid, paths=5000
    )
    # With zero jumps, behaves like CIR; terminal mean close to theta.
    np.testing.assert_allclose(paths[:, -1].mean(), 0.05, rtol=1e-1)
