# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Cox-Ingersoll-Ross (1985) short-rate utilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dxquant._validation import positive_integer
from dxquant.time import uniform_time_grid

__all__ = ["cir_zero_coupon", "CIRShortRateProcess", "calibrate_cir"]


def _gamma(kappa: float, sigma: float) -> float:
    return math.sqrt(kappa**2 + 2 * sigma**2)


def cir_zero_coupon(
    short_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    maturity: float,
) -> float:
    """
    Returns the analytical zero-coupon bond price in the CIR model.
    """

    values = (short_rate, kappa, theta, sigma, maturity)
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("CIR inputs must be finite and nonnegative")
    if maturity == 0:
        return 1.0
    if sigma == 0:
        duration = (
            maturity if kappa == 0 else -math.expm1(-kappa * maturity) / kappa
        )
        integral = theta * maturity + (short_rate - theta) * duration
        return math.exp(-integral)
    g = _gamma(kappa, sigma)
    one_minus_decay = -math.expm1(-g * maturity)
    denominator = 2 * g - 2 * sigma**2 * one_minus_decay / (kappa + g)
    b = 2 * one_minus_decay / denominator
    log_a = (
        2
        * kappa
        * theta
        * (
            -maturity / (kappa + g)
            - math.log1p(-(sigma**2) * one_minus_decay / (g * (kappa + g)))
            / sigma**2
        )
    )
    return math.exp(log_a - b * short_rate)


@dataclass
class CIRShortRateProcess:
    """
    Euler-Maruyama simulation for the CIR short-rate process.
    """

    kappa: float
    theta: float
    sigma: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(x) and x >= 0
            for x in (self.kappa, self.theta, self.sigma)
        ):
            raise ValueError("CIR parameters must be finite and nonnegative")

    def simulate_paths(
        self,
        short_rate: float,
        maturity: float,
        steps: int,
        paths: int,
    ) -> np.ndarray:
        """
        Sample projected Euler rates or exact deterministic zero-sigma paths.
        """

        positive_integer(paths, "paths")
        if not math.isfinite(short_rate) or short_rate < 0:
            raise ValueError("short_rate must be finite and nonnegative")
        grid = uniform_time_grid(maturity, steps)
        rng = np.random.default_rng(self.seed)
        result = np.empty((paths, grid.size), dtype=float)
        result[:, 0] = short_rate
        for idx, dt in enumerate(np.diff(grid)):
            z = rng.standard_normal(paths)
            prev = result[:, idx]
            if self.sigma == 0:
                result[:, idx + 1] = self.theta + (
                    prev - self.theta
                ) * math.exp(-self.kappa * dt)
                continue
            drift = self.kappa * (self.theta - prev) * dt
            diffusion = (
                self.sigma * np.sqrt(np.maximum(prev, 0.0)) * np.sqrt(dt) * z
            )
            result[:, idx + 1] = np.maximum(prev + drift + diffusion, 0.0)
        return result


def calibrate_cir(
    initial_rate: float,
    maturities: Sequence[float],
    target_prices: Sequence[float],
    initial_guess: Sequence[float]=(0.5, 0.05, 0.1),
) -> np.ndarray:
    """
    Calibrate CIR parameters (kappa, theta, sigma) to target zero-coupon prices.
    """

    maturity_arr = np.asarray(maturities, dtype=float)
    target_arr = np.asarray(target_prices, dtype=float)

    def model(params: np.ndarray) -> np.ndarray:
        kappa, theta, sigma = params
        return np.array(
            [
                cir_zero_coupon(initial_rate, kappa, theta, sigma, t)
                for t in maturity_arr
            ]
        )

    from dxquant.calibration import calibrate_least_squares

    bounds = ([1e-4, 1e-4, 1e-4], [5.0, 0.2, 1.0])
    return calibrate_least_squares(
        model, target_arr.tolist(), initial_guess, bounds=bounds
    )
