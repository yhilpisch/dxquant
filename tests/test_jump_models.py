"""Independent compound-Poisson benchmarks for the jump processes."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from dxquant.processes import JumpDiffusionProcess, SquareRootJumpDiffusion


def _assert_moments(
    samples: np.ndarray,
    mean: float,
    variance: float,
    fourth_central: float,
) -> None:
    """Use six standard errors for IID sample mean/variance comparisons."""

    count = samples.size
    mean_error = 6 * math.sqrt(variance / count)
    variance_error = 6 * math.sqrt((fourth_central - variance**2) / count)
    assert abs(float(samples.mean()) - mean) <= mean_error + 1e-12
    assert abs(float(samples.var(ddof=1)) - variance) <= (
        variance_error + 1e-12
    )


def test_merton_multiple_downward_jumps_remain_positive() -> None:
    process = JumpDiffusionProcess(0.02, 0, 4, -1, 0, seed=1)
    terminal = process.simulate_paths(100, [0, 1], 100)[:, -1]
    assert np.all(terminal > 0)
    compensation = 4 * math.expm1(-1)
    counts = (np.log(terminal / 100) - 0.02 + compensation) / -1
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-12)
    assert np.all(counts >= 0)
    assert np.any(counts > 1)


@pytest.mark.parametrize("steps", [1, 8, 64])
def test_merton_terminal_log_moments_across_grids(steps: int) -> None:
    process = JumpDiffusionProcess(
        0.02,
        0.3,
        4,
        -0.15,
        0.45,
        seed=17,
        antithetic=False,
        moment_matching=False,
    )
    terminal = process.simulate_paths(100, np.linspace(0, 1, steps + 1), 50000)[
        :, -1
    ]
    mu, sigma, intensity = -0.15, 0.45, 4
    compensation = intensity * math.expm1(mu + 0.5 * sigma**2)
    mean = 0.02 - compensation - 0.5 * 0.3**2 + intensity * mu
    variance = 0.3**2 + intensity * (mu**2 + sigma**2)
    fourth_jump = mu**4 + 6 * mu**2 * sigma**2 + 3 * sigma**4
    fourth_central = intensity * fourth_jump + 3 * variance**2
    _assert_moments(np.log(terminal / 100), mean, variance, fourth_central)
    for frequency in (0.5, 1.0, 2.0):
        jump_cf = np.exp(1j * frequency * mu - 0.5 * sigma**2 * frequency**2)
        expected = np.exp(
            1j * frequency * (mean - intensity * mu)
            - 0.5 * 0.3**2 * frequency**2
            + intensity * (jump_cf - 1)
        )
        empirical = np.mean(np.exp(1j * frequency * np.log(terminal / 100)))
        assert abs(empirical - expected) < 6 / math.sqrt(terminal.size)


def _merton_call_series(intensity: float) -> float:
    """Poisson mixture of conditional lognormal call expectations at T=1."""

    spot, strike, rate, vol = 100.0, 100.0, 0.02, 0.2
    mu, sigma = -0.15, 0.25
    correction = intensity * math.expm1(mu + 0.5 * sigma**2)
    weight = math.exp(-intensity)
    price = 0.0
    for count in range(80):
        mean = math.log(spot) + rate - correction - vol**2 / 2 + count * mu
        variance = vol**2 + count * sigma**2
        deviation = math.sqrt(variance)
        d2 = (mean - math.log(strike)) / deviation
        conditional = math.exp(mean + variance / 2) * ndtr(
            d2 + deviation
        ) - strike * ndtr(d2)
        price += weight * conditional
        weight *= intensity / (count + 1)
    return float(math.exp(-rate) * price)


@pytest.mark.parametrize("intensity", [0.5, 4.0])
@pytest.mark.parametrize("steps", [1, 32])
def test_merton_call_matches_independent_series(
    intensity: float,
    steps: int,
) -> None:
    process = JumpDiffusionProcess(
        0.02,
        0.2,
        intensity,
        -0.15,
        0.25,
        seed=23,
        antithetic=False,
        moment_matching=False,
    )
    terminal = process.simulate_paths(100, np.linspace(0, 1, steps + 1), 80000)[
        :, -1
    ]
    payoffs = math.exp(-0.02) * np.maximum(terminal - 100, 0)
    error = 6 * payoffs.std(ddof=1) / math.sqrt(payoffs.size)
    assert abs(payoffs.mean() - _merton_call_series(intensity)) <= error
    correction = math.expm1(-0.15 + 0.25**2 / 2)
    mean = 100 * math.exp(0.02)
    second = 100**2 * math.exp(
        2 * 0.02
        + 0.2**2
        + intensity * (math.expm1(-0.3 + 2 * 0.25**2) - 2 * correction)
    )
    mean_error = 6 * math.sqrt((second - mean**2) / terminal.size)
    assert abs(terminal.mean() - mean) <= mean_error


@pytest.mark.parametrize("intensity", [0.0, 0.5, 4.0])
@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_additive_square_root_jump_moments(
    intensity: float,
    sigma: float,
) -> None:
    # A large constant level avoids clipping; one step isolates the jump law.
    process = SquareRootJumpDiffusion(
        0.8,
        100,
        0,
        intensity,
        0.1,
        sigma,
        seed=29,
        antithetic=False,
        moment_matching=False,
    )
    samples = process.simulate_paths(100, [0, 1], 80000)[:, -1] - 100
    first = math.expm1(0.1 + sigma**2 / 2)
    second = math.exp(0.2 + 2 * sigma**2) - 2 * math.exp(0.1 + sigma**2 / 2) + 1
    fourth = sum(
        math.comb(4, k)
        * (-1) ** (4 - k)
        * math.exp(k * 0.1 + k**2 * sigma**2 / 2)
        for k in range(5)
    )
    variance = intensity * second
    _assert_moments(
        samples,
        intensity * first,
        variance,
        intensity * fourth + 3 * variance**2,
    )


def test_square_root_negative_marks_keep_projection_at_zero() -> None:
    process = SquareRootJumpDiffusion(1, 0.04, 0, 4, -1, 0, seed=31)
    terminal = process.simulate_paths(0.04, [0, 1], 30000)[:, -1]
    assert set(np.unique(terminal)) == {0.0, 0.04}
    # A single downward mark already crosses zero; only N=0 keeps the level.
    probability = math.exp(-4)
    observed = np.mean(terminal == 0.04)
    error = 6 * math.sqrt(probability * (1 - probability) / terminal.size)
    assert abs(observed - probability) < error


@pytest.mark.parametrize("model", ["merton", "square-root"])
def test_jump_and_diffusion_innovations_are_uncorrelated(model: str) -> None:
    if model == "merton":
        process = JumpDiffusionProcess(
            0.02,
            0.2,
            4,
            0.2,
            0.05,
            seed=37,
            antithetic=False,
            moment_matching=False,
        )
        full = process.simulate_paths(100, [0, 1], 40000)[:, -1]
        pure_jump = replace(process, volatility=0).simulate_paths(
            100, [0, 1], 40000
        )[:, -1]
        diffusion = np.log(full / pure_jump) + 0.2**2 / 2
        jumps = np.log(pure_jump)
    else:
        square_root = SquareRootJumpDiffusion(
            1,
            100,
            0.2,
            4,
            0.2,
            0.05,
            seed=37,
            antithetic=False,
            moment_matching=False,
        )
        full = square_root.simulate_paths(100, [0, 1], 40000)[:, -1]
        pure_jump = replace(square_root, volatility=0).simulate_paths(
            100, [0, 1], 40000
        )[:, -1]
        diffusion = full - pure_jump
        jumps = pure_jump - 100
    assert abs(np.corrcoef(diffusion, jumps)[0, 1]) < 6 / math.sqrt(40000)


@pytest.mark.parametrize("model", ["merton", "square-root"])
def test_jump_processes_replay_seeded_paths(model: str) -> None:
    if model == "merton":
        process = JumpDiffusionProcess(0.02, 0.2, 4, -0.1, 0.3, seed=41)
        first = process.simulate_paths(100, [0, 0.25, 1], 100)
        second = process.simulate_paths(100, [0, 0.25, 1], 100)
        other = replace(process, seed=42).simulate_paths(100, [0, 0.25, 1], 100)
    else:
        square_root = SquareRootJumpDiffusion(
            1, 0.04, 0.2, 4, -0.1, 0.3, seed=41
        )
        first = square_root.simulate_paths(0.04, [0, 0.25, 1], 100)
        second = square_root.simulate_paths(0.04, [0, 0.25, 1], 100)
        other = replace(square_root, seed=42).simulate_paths(
            0.04, [0, 0.25, 1], 100
        )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
