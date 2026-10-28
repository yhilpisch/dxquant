# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Stochastic process abstractions for Monte Carlo simulations.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._validation import finite_parameters, positive_integer, simulation_grid
from .random import sn_random_numbers
from .time import uniform_time_grid

FloatArray = NDArray[np.float64]
TimeGrid = Sequence[float] | FloatArray


def _jump_random_streams(
    seed: int | None,
) -> tuple[int, np.random.Generator, np.random.Generator]:
    """Separate diffusion, arrival-count, and jump-size random streams."""

    diffusion, counts, sizes = np.random.SeedSequence(seed).spawn(3)
    normal_seed = int(diffusion.generate_state(1, dtype=np.uint64)[0])
    return (
        normal_seed,
        np.random.default_rng(counts),
        np.random.default_rng(sizes),
    )


@dataclass(frozen=True)
class _DeterministicLogPrice:
    """Callable log-price characteristic function with exact point mass."""

    log_price: float

    def __call__(self, u: complex) -> complex:
        return cmath.exp(1j * u * self.log_price)


__all__ = [
    "GeometricBrownianMotion",
    "MultiAssetGeometricBrownianMotion",
    "HestonProcess",
    "JumpDiffusionProcess",
    "MeanRevertingDiffusion",
    "SquareRootJumpDiffusion",
    "SABRProcess",
    "sabr_implied_volatility",
]


@dataclass
class GeometricBrownianMotion:
    """
    Log-normal diffusion with analytic Euler discretization.
    """

    drift: float
    volatility: float
    antithetic: bool = True
    moment_matching: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        finite_parameters(drift=self.drift, volatility=self.volatility)
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    def simulate_paths(
        self,
        spot: float,
        time_grid: TimeGrid,
        paths: int,
    ) -> FloatArray:
        """
        Simulate GBM paths sampled along ``time_grid`` (year fractions).
        """

        grid = simulation_grid(time_grid, paths)
        if not math.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be positive for GBM")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

        n_steps = grid.size - 1
        increments = sn_random_numbers(
            (n_steps, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=self.seed,
        )

        dt = np.diff(grid)
        paths_arr = np.empty((paths, grid.size), dtype=float)
        paths_arr[:, 0] = spot

        for idx, step in enumerate(dt):
            drift_term = (self.drift - 0.5 * self.volatility**2) * step
            diffusion_term = self.volatility * np.sqrt(step) * increments[idx]
            paths_arr[:, idx + 1] = paths_arr[:, idx] * np.exp(
                drift_term + diffusion_term
            )

        return paths_arr


@dataclass
class MultiAssetGeometricBrownianMotion:
    """
    Vector-valued GBM with constant correlations.
    """

    drifts: Sequence[float]
    volatilities: Sequence[float]
    correlation: np.ndarray
    antithetic: bool = True
    moment_matching: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        self._drifts = np.asarray(self.drifts, dtype=float)
        self._vols = np.asarray(self.volatilities, dtype=float)
        if self._drifts.ndim != 1 or not self._drifts.size:
            raise ValueError("drifts must be a nonempty vector")
        if not np.all(np.isfinite(self._drifts)) or not np.all(
            np.isfinite(self._vols)
        ):
            raise ValueError("drifts and volatilities must be finite")
        if self._drifts.shape != self._vols.shape:
            raise ValueError(
                "drifts and volatilities must share the same dimension"
            )
        if np.any(self._vols < 0):
            raise ValueError("volatilities must be non-negative")
        self._dimension = self._drifts.size
        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (self._dimension, self._dimension):
            raise ValueError(
                "correlation matrix must be square with matching dimension"
            )
        if not np.allclose(corr, corr.T):
            raise ValueError("correlation matrix must be symmetric")
        if not np.all(np.isfinite(corr)) or not np.allclose(np.diag(corr), 1):
            raise ValueError("correlation must be finite with unit diagonal")
        try:
            self._chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation must be positive definite") from exc

    def simulate_paths(
        self,
        spot: Sequence[float] | FloatArray,
        time_grid: TimeGrid,
        paths: int,
    ) -> FloatArray:
        """
        Simulate correlated lognormal paths with a positive-definite
        correlation.
        """

        grid = simulation_grid(time_grid, paths)

        spot_arr = np.asarray(spot, dtype=float)
        if spot_arr.shape != (self._dimension,):
            raise ValueError("spots vector must match drift dimension")
        if not np.all(np.isfinite(spot_arr)) or np.any(spot_arr <= 0):
            raise ValueError("spots must be strictly positive")

        n_steps = grid.size - 1
        normals_raw = sn_random_numbers(
            (n_steps, self._dimension, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=self.seed,
        )
        normals = np.swapaxes(
            normals_raw, 1, 2
        )  # shape -> (n_steps, paths, dim)

        dt = np.diff(grid)
        result = np.empty((paths, grid.size, self._dimension), dtype=float)
        result[:, 0, :] = spot_arr

        drift_vec = self._drifts - 0.5 * self._vols**2

        for idx, step in enumerate(dt):
            correlated = normals[idx] @ self._chol.T  # shape (paths, dim)
            diffusion = correlated * (self._vols * np.sqrt(step))
            drift = drift_vec * step
            increment = np.exp(drift + diffusion)
            result[:, idx + 1, :] = result[:, idx, :] * increment

        return result


@dataclass
class HestonProcess:
    """
    Stochastic volatility process following the classic Heston (1993)
    specification.
    """

    kappa: float  # mean reversion speed
    theta: float  # long-term variance
    sigma: float  # vol-of-vol
    rho: float  # correlation between Brownian motions
    drift: float
    antithetic: bool = True
    moment_matching: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(x)
            for x in (self.kappa, self.theta, self.sigma, self.rho, self.drift)
        ):
            raise ValueError("Heston parameters must be finite")
        if self.sigma < 0 or self.theta < 0 or self.kappa <= 0:
            raise ValueError("Heston parameters must be positive")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")

    def simulate_paths(
        self,
        spot: float,
        variance: float,
        time_grid: TimeGrid,
        paths: int,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Projected Euler variance and log spot using previous variance.
        """

        grid = simulation_grid(time_grid, paths)
        if not math.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be positive")
        if not math.isfinite(variance) or variance < 0:
            raise ValueError("variance must be non-negative")

        n_steps = grid.size - 1
        normals = sn_random_numbers(
            (n_steps, 2, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=self.seed,
        )
        normals = np.swapaxes(normals, 1, 2)  # (steps, paths, 2)

        paths_spot = np.empty((paths, grid.size), dtype=float)
        paths_var = np.empty_like(paths_spot)
        paths_spot[:, 0] = spot
        paths_var[:, 0] = variance

        for idx, step in enumerate(np.diff(grid)):
            z1 = normals[idx, :, 0]
            z2 = normals[idx, :, 1]
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            prev_var = paths_var[:, idx]
            sqrt_var = np.sqrt(np.maximum(prev_var, 0.0))
            drift_var = self.kappa * (self.theta - prev_var) * step
            diffusion_var = self.sigma * sqrt_var * np.sqrt(step) * w2
            new_var = np.maximum(prev_var + drift_var + diffusion_var, 0.0)
            paths_var[:, idx + 1] = new_var

            if self.sigma == 0:
                decay = math.exp(-self.kappa * step)
                paths_var[:, idx + 1] = (
                    self.theta + (prev_var - self.theta) * decay
                )
                integrated = (
                    self.theta * step
                    + (prev_var - self.theta)
                    * (-math.expm1(-self.kappa * step))
                    / self.kappa
                )
                drift = self.drift * step - 0.5 * integrated
                diffusion = np.sqrt(np.maximum(integrated, 0.0)) * w1
            else:
                drift = (self.drift - 0.5 * prev_var) * step
                diffusion = sqrt_var * np.sqrt(step) * w1
            paths_spot[:, idx + 1] = paths_spot[:, idx] * np.exp(
                drift + diffusion
            )

        return paths_spot, paths_var

    def characteristic_function(
        self, tau: float, spot: float, variance: float
    ) -> Callable[[complex], complex]:
        """
        Heston characteristic function as per original paper.
        """

        if not all(math.isfinite(x) for x in (tau, spot, variance)):
            raise ValueError("maturity, spot, and variance must be finite")
        if tau < 0 or spot <= 0 or variance < 0:
            raise ValueError("invalid characteristic-function domain")
        integrated = (
            self.theta * tau
            + (variance - self.theta)
            * (-math.expm1(-self.kappa * tau))
            / self.kappa
        )

        if tau == 0 or (variance == 0 and self.theta == 0):
            return _DeterministicLogPrice(math.log(spot) + self.drift * tau)

        def phi(u: complex) -> complex:
            if tau == 0:
                return cmath.exp(1j * u * math.log(spot))
            if self.sigma == 0:
                return cmath.exp(
                    1j * u * (math.log(spot) + self.drift * tau)
                    - 0.5 * (u * u + 1j * u) * integrated
                )
            if u == 0:
                return 1.0 + 0j
            if u == -1j:
                return complex(spot * math.exp(self.drift * tau))
            q = u * u + 1j * u
            base = self.kappa - self.rho * self.sigma * 1j * u
            d = cmath.sqrt(base * base + self.sigma**2 * q)
            # Rationalization avoids cancellation as vol-of-vol tends to zero.
            scale = -q / (base + d)
            g = self.sigma**2 * scale / (base + d)
            decay = cmath.exp(-d * tau)
            ratio = g * (1 - decay) / (1 - g)
            if abs(ratio) < 1e-4:
                log_ratio = ratio * (
                    1 - ratio / 2 + ratio**2 / 3 - ratio**3 / 4
                )
            else:
                log_ratio = cmath.log(1 + ratio)
            c = (
                self.kappa
                * self.theta
                * (scale * tau - 2 * log_ratio / self.sigma**2)
            )
            v = scale * (1 - decay) / (1 - g * decay)
            return cmath.exp(
                1j * u * (math.log(spot) + self.drift * tau) + c + variance * v
            )

        return phi


@dataclass
class JumpDiffusionProcess:
    """
    Merton jump-diffusion with independent normal log-jump sizes.

    Given N arrivals in a step, the total log jump is normal with mean
    N * jump_mean and variance N * jump_std**2. The exponential update
    compounds every arrival and includes the expected-jump drift correction.
    Antithetic/moment matching options apply only to diffusion draws.
    Seeded calls replay the same paths using separate child random streams.
    """

    drift: float
    volatility: float
    jump_intensity: float  # lambda
    jump_mean: float
    jump_std: float
    seed: int | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        finite_parameters(
            drift=self.drift,
            volatility=self.volatility,
            jump_intensity=self.jump_intensity,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
        )
        if self.volatility < 0 or self.jump_intensity < 0 or self.jump_std < 0:
            raise ValueError(
                "Volatility, jump intensity, and jump std must be non-negative"
            )

    def simulate_paths(
        self,
        spot: float,
        time_grid: TimeGrid,
        paths: int,
    ) -> FloatArray:
        """
        Sample compensated Merton increments with every arrival compounded.
        """

        grid = simulation_grid(time_grid, paths)

        normal_seed, count_rng, jump_rng = _jump_random_streams(self.seed)
        n_steps = grid.size - 1
        normals = sn_random_numbers(
            (n_steps, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=normal_seed,
        )

        result = np.empty((paths, grid.size), dtype=float)
        result[:, 0] = spot
        jump_compensation = self.jump_intensity * (
            math.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1
        )

        for idx, step in enumerate(np.diff(grid)):
            poisson = count_rng.poisson(self.jump_intensity * step, size=paths)
            jump_normal = jump_rng.standard_normal(paths)
            log_jumps = (
                poisson * self.jump_mean
                + np.sqrt(poisson) * self.jump_std * jump_normal
            )
            diffusion = self.volatility * np.sqrt(step) * normals[idx]
            drift = (
                self.drift - jump_compensation - 0.5 * self.volatility**2
            ) * step
            result[:, idx + 1] = result[:, idx] * np.exp(
                drift + diffusion + log_jumps
            )

        return result


@dataclass
class MeanRevertingDiffusion:
    """
    Mean-reverting diffusion with proportional (multiplicative) volatility.

    Discretization: ``x_{n+1} = x_n + kappa*(theta - x_n)*dt +
    x_n*sigma*sqrt(dt)*dW``.
    This mirrors ``dx.models.mean_reverting_diffusion`` (a Vasicek-style process
    with
    level-proportional volatility, often used for volatilities or commodities).
    """

    kappa: float  # mean reversion speed
    theta: float  # long-run mean
    volatility: float
    antithetic: bool = True
    moment_matching: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        finite_parameters(
            kappa=self.kappa, theta=self.theta, volatility=self.volatility
        )
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    def simulate_paths(
        self,
        initial_value: float,
        time_grid: TimeGrid,
        paths: int,
    ) -> FloatArray:
        """Apply Euler mean reversion with level-proportional diffusion."""

        grid = simulation_grid(time_grid, paths)
        if not math.isfinite(initial_value) or initial_value <= 0:
            raise ValueError("initial_value must be positive for this process")

        n_steps = grid.size - 1
        normals = sn_random_numbers(
            (n_steps, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=self.seed,
        )

        result = np.empty((paths, grid.size), dtype=float)
        result[:, 0] = initial_value
        for idx, step in enumerate(np.diff(grid)):
            prev = result[:, idx]
            drift = self.kappa * (self.theta - prev) * step  # pull toward theta
            diffusion = prev * self.volatility * np.sqrt(step) * normals[idx]
            result[:, idx + 1] = prev + drift + diffusion
        return result


@dataclass
class SquareRootJumpDiffusion:
    """
    Projected Euler square-root diffusion with additive signed jumps.

    Each arrival adds exp(Y) - 1, where Y is normal with parameters
    jump_mean/jump_std. Sum independent marks for all arrivals, add the
    Euler diffusion update, then clip the resulting level at zero. There
    is no jump drift compensator. This preserves DXQuant's additive convention,
    which differs from the legacy level-scaled jump implementation.
    Antithetic/moment matching options affect diffusion draws only.
    """

    kappa: float
    theta: float
    volatility: float
    jump_intensity: float  # lambda
    jump_mean: float  # mean log-jump size
    jump_std: float  # std dev of log-jump size
    seed: int | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        finite_parameters(
            kappa=self.kappa,
            theta=self.theta,
            volatility=self.volatility,
            jump_intensity=self.jump_intensity,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
        )
        if self.kappa <= 0 or self.theta < 0:
            raise ValueError("kappa must be positive and theta non-negative")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.jump_intensity < 0 or self.jump_std < 0:
            raise ValueError("jump intensity and jump std must be non-negative")

    def simulate_paths(
        self,
        initial_value: float,
        time_grid: TimeGrid,
        paths: int,
    ) -> FloatArray:
        """Sum independent additive marks and project Euler levels at zero."""

        grid = simulation_grid(time_grid, paths)
        if not math.isfinite(initial_value) or initial_value < 0:
            raise ValueError("initial_value must be non-negative")

        normal_seed, count_rng, jump_rng = _jump_random_streams(self.seed)
        n_steps = grid.size - 1
        normals = sn_random_numbers(
            (n_steps, paths),
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
            seed=normal_seed,
        )

        result = np.empty((paths, grid.size), dtype=float)
        result[:, 0] = initial_value
        for idx, step in enumerate(np.diff(grid)):
            prev = result[:, idx]
            prev_trunc = np.maximum(prev, 0.0)  # full truncation
            drift = self.kappa * (self.theta - prev_trunc) * step
            diffusion = (
                np.sqrt(prev_trunc)
                * self.volatility
                * np.sqrt(step)
                * normals[idx]
            )
            poisson = count_rng.poisson(self.jump_intensity * step, size=paths)
            owners = np.repeat(np.arange(paths), poisson)
            log_marks = jump_rng.normal(
                self.jump_mean, self.jump_std, size=owners.size
            )
            jumps = np.bincount(
                owners, weights=np.expm1(log_marks), minlength=paths
            )
            result[:, idx + 1] = np.maximum(
                prev + drift + diffusion + jumps, 0.0
            )
        return result


@dataclass
class SABRProcess:
    """SABR volatility diffusion with a frozen-coefficient forward update.

    Volatility follows d(alpha) = nu * alpha * dZ and is sampled exactly.
    Forward increments use the previous volatility, with correlation rho.
    Beta=1 uses a log update; beta=0 permits signed normal forwards;
    intermediate beta uses Euler with an absorbing zero boundary.
    """

    initial_value: float
    alpha: float
    beta: float
    rho: float
    vol_vol: float
    drift: float
    seed: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.initial_value,
            self.alpha,
            self.beta,
            self.rho,
            self.vol_vol,
            self.drift,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("SABR parameters must be finite")
        if self.alpha < 0 or self.vol_vol < 0 or not 0 <= self.beta <= 1:
            raise ValueError("invalid SABR volatility/beta parameters")
        if not -1 <= self.rho <= 1:
            raise ValueError("rho must be in [-1, 1]")
        if self.beta > 0 and self.initial_value <= 0:
            raise ValueError("positive initial value required for beta > 0")

    def simulate_paths(
        self, maturity: float, steps: int, paths: int
    ) -> np.ndarray:
        """
        Return forward paths; volatility updates use independent past data.
        """

        positive_integer(paths, "paths")
        grid = uniform_time_grid(maturity, steps)
        spot_paths = np.empty((paths, grid.size), dtype=float)
        spot_paths[:, 0] = self.initial_value
        alpha = np.full(paths, self.alpha, dtype=float)
        normals = sn_random_numbers((steps, 2, paths), seed=self.seed)
        for idx, step in enumerate(np.diff(grid)):
            z1, z2 = normals[idx]
            correlated = self.rho * z1 + math.sqrt(1 - self.rho**2) * z2
            previous = spot_paths[:, idx]
            if self.beta == 1:
                updated = previous * np.exp(
                    (self.drift - 0.5 * alpha**2) * step
                    + alpha * math.sqrt(step) * z1
                )
            else:
                level = previous if self.beta == 0 else np.maximum(previous, 0)
                updated = (
                    previous
                    + self.drift * previous * step
                    + alpha * np.power(level, self.beta) * math.sqrt(step) * z1
                )
                if self.beta > 0:
                    updated = np.maximum(updated, 0)
            spot_paths[:, idx + 1] = updated
            alpha *= np.exp(
                -0.5 * self.vol_vol**2 * step
                + self.vol_vol * math.sqrt(step) * correlated
            )
        return spot_paths


def sabr_implied_volatility(
    forward: float,
    strike: float,
    maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    vol_vol: float,
) -> float:
    """Hagan lognormal SABR approximation with the stable small-z limit.

    Requires positive forward/strike/alpha, beta in [0, 1], abs(rho) < 1,
    and nonnegative maturity/vol-of-vol. This is an asymptotic approximation.
    """

    values = (forward, strike, maturity, alpha, beta, rho, vol_vol)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("SABR inputs must be finite")
    if forward <= 0 or strike <= 0:
        raise ValueError(
            "Forward and strike must be positive for SABR approximation"
        )
    if alpha <= 0 or vol_vol < 0 or maturity < 0:
        raise ValueError("invalid SABR alpha, vol-of-vol, or maturity")
    if not 0 <= beta <= 1 or not -1 < rho < 1:
        raise ValueError("beta must be in [0, 1] and abs(rho) < 1")
    complement = 1 - beta
    power = (forward * strike) ** (complement / 2)
    log_moneyness = math.log(forward / strike)
    z = vol_vol * power * log_moneyness / alpha
    if abs(z) < 1e-6:
        ratio = 1 - rho * z / 2 + (2 - 3 * rho**2) * z**2 / 12
    else:
        # asinh differences avoid subtracting nearly equal tail quantities.
        scale = math.sqrt(1 - rho**2)
        x = math.asinh((z - rho) / scale) + math.asinh(rho / scale)
        ratio = z / x
    log_term = complement**2 * log_moneyness**2
    denominator = power * (1 + log_term / 24 + log_term**2 / 1920)
    correction = (
        complement**2 * alpha**2 / (24 * power**2)
        + rho * beta * vol_vol * alpha / (4 * power)
        + (2 - 3 * rho**2) * vol_vol**2 / 24
    )
    result = alpha * ratio / denominator * (1 + correction * maturity)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            "SABR approximation outside supported parameter regime"
        )
    return float(result)
