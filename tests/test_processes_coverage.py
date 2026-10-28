import numpy as np
import pytest

from dxquant.processes import (
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
    SABRProcess,
    sabr_implied_volatility,
)


def test_gbm_errors() -> None:
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        GeometricBrownianMotion(0.01, 0.2).simulate_paths(
            100.0, np.array([[0.0, 1.0], [0.0, 2.0]]), 1
        )
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        GeometricBrownianMotion(0.01, 0.2).simulate_paths(
            100.0, np.array([1.0, 2.0]), 1
        )
    with pytest.raises(
        ValueError, match="time_grid must be strictly increasing"
    ):
        GeometricBrownianMotion(0.01, 0.2).simulate_paths(
            100.0, np.array([0.0, 0.0, 1.0]), 1
        )
    with pytest.raises(ValueError, match="spot must be positive for GBM"):
        GeometricBrownianMotion(0.01, 0.2).simulate_paths(
            -100.0, np.array([0.0, 1.0]), 1
        )
    with pytest.raises(ValueError, match="volatility must be non-negative"):
        GeometricBrownianMotion(0.01, -0.2).simulate_paths(
            100.0, np.array([0.0, 1.0]), 1
        )


def test_heston_errors() -> None:
    with pytest.raises(ValueError, match="Heston parameters must be positive"):
        HestonProcess(-0.5, 0.04, 0.1, 0.5, 0.02)
    with pytest.raises(ValueError, match=r"rho must be in \[-1, 1\]"):
        HestonProcess(0.5, 0.04, 0.1, 1.5, 0.02)
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        HestonProcess(0.5, 0.04, 0.1, 0.5, 0.02).simulate_paths(
            100, 0.04, np.array([[0.0, 1.0], [0.0, 2.0]]), 1
        )
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        HestonProcess(0.5, 0.04, 0.1, 0.5, 0.02).simulate_paths(
            100, 0.04, np.array([1.0, 2.0]), 1
        )
    with pytest.raises(
        ValueError, match="time_grid must be strictly increasing"
    ):
        HestonProcess(0.5, 0.04, 0.1, 0.5, 0.02).simulate_paths(
            100, 0.04, np.array([0.0, 0.0, 1.0]), 1
        )
    with pytest.raises(ValueError, match="spot must be positive"):
        HestonProcess(0.5, 0.04, 0.1, 0.5, 0.02).simulate_paths(
            -100, 0.04, np.array([0.0, 1.0]), 1
        )
    with pytest.raises(ValueError, match="variance must be non-negative"):
        HestonProcess(0.5, 0.04, 0.1, 0.5, 0.02).simulate_paths(
            100, -0.04, np.array([0.0, 1.0]), 1
        )


def test_jump_diffusion_errors() -> None:
    with pytest.raises(
        ValueError,
        match="Volatility, jump intensity, and jump std must be non-negative",
    ):
        JumpDiffusionProcess(0.01, -0.2, 0.2, 0.1, 0.2)
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        JumpDiffusionProcess(0.01, 0.2, 0.2, 0.1, 0.2).simulate_paths(
            100, np.array([[0.0, 1.0], [0.0, 2.0]]), 1
        )
    with pytest.raises(ValueError, match="time_grid must be 1D and start at 0"):
        JumpDiffusionProcess(0.01, 0.2, 0.2, 0.1, 0.2).simulate_paths(
            100, np.array([1.0, 2.0]), 1
        )
    with pytest.raises(
        ValueError, match="time_grid must be strictly increasing"
    ):
        JumpDiffusionProcess(0.01, 0.2, 0.2, 0.1, 0.2).simulate_paths(
            100, np.array([0.0, 0.0, 1.0]), 1
        )


def test_sabr_process_output_shape() -> None:
    sabr = SABRProcess(100, 0.2, 0.5, 0.5, 0.1, 0.02)
    paths = sabr.simulate_paths(1.0, 10, 5)
    assert paths.shape == (5, 11)


def test_sabr_implied_volatility_errors() -> None:
    with pytest.raises(
        ValueError,
        match="Forward and strike must be positive for SABR approximation",
    ):
        sabr_implied_volatility(-100, 100, 1.0, 0.2, 0.5, 0.5, 0.1)
    # test the case where forward and strike are very close
    assert sabr_implied_volatility(100, 100, 1.0, 0.2, 0.5, 0.5, 0.1) > 0
    assert sabr_implied_volatility(100, 110, 1.0, 0.2, 0.5, 0.5, 0.1) > 0
