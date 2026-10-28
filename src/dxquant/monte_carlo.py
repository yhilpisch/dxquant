# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Monte Carlo valuation harness.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import sqrt

import numpy as np

from ._validation import positive_integer
from .env import MarketEnvironment
from .payoffs import PathPayoff, TerminalPayoff
from .processes import (
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
    MultiAssetGeometricBrownianMotion,
)
from .time import DayCount, uniform_time_grid, year_fraction

__all__ = ["MonteCarloPricer", "MonteCarloResult"]


def _sampling_error(samples: np.ndarray, scheme: str) -> float:
    """Estimate mean error only for supported independent sampling units."""

    if scheme == "deterministic":
        return 0.0
    if samples.size < 2 or scheme not in ("iid", "antithetic"):
        return float("nan")
    if scheme == "antithetic":
        if samples.size % 2 or samples.size < 4:
            return float("nan")
        half = samples.size // 2
        samples = (samples[:half] + samples[half:]) / 2
    return float(np.std(samples, ddof=1) / sqrt(samples.size))


@dataclass
class MonteCarloResult:
    """Pricing samples and uncertainty metadata.

    std_error uses IID paths or independent antithetic pair means. It is NaN
    for a single moment-matched batch or unsupported sampling units; estimate
    those errors from independently seeded whole-batch replications instead.
    """

    price: float
    std_error: float
    discounted_payoffs: np.ndarray
    sampling_scheme: str = "unknown"

    def percentile(self, q: float) -> float:
        """Return percentile of discounted payoffs for risk reporting."""

        if not np.isfinite(q) or not 0 <= q <= 100:
            raise ValueError("percentile must be in [0, 100]")
        return float(np.percentile(self.discounted_payoffs, q))


@dataclass
class MonteCarloPricer:
    """
    Coordinates process simulation, payoff evaluation, and discounting.
    """

    env: MarketEnvironment
    process: (
        GeometricBrownianMotion
        | HestonProcess
        | MultiAssetGeometricBrownianMotion
        | JumpDiffusionProcess
    )
    discount_curve_key: str
    paths: int = 50000
    steps: int = 64
    day_count: DayCount | float = DayCount.ACT_365
    initial_variance: float | None = None

    def __post_init__(self) -> None:
        positive_integer(self.paths, "paths")
        positive_integer(self.steps, "steps")
        year_fraction(
            self.env.pricing_date, self.env.pricing_date, self.day_count
        )
        if self.initial_variance is not None and (
            not np.isfinite(self.initial_variance) or self.initial_variance < 0
        ):
            raise ValueError("initial_variance must be finite and nonnegative")

    def _prepare_grid(
        self, maturity: dt.date | dt.datetime
    ) -> tuple[np.ndarray, float]:
        maturity_years = year_fraction(
            self.env.pricing_date, maturity, self.day_count
        )
        grid = (
            np.array([0.0])
            if maturity_years == 0
            else uniform_time_grid(maturity_years, self.steps)
        )
        return grid, maturity_years

    def _discount(
        self, maturity: dt.date | dt.datetime, payoffs: np.ndarray
    ) -> np.ndarray:
        discount_curve = self.env.get_curve(self.discount_curve_key)
        discount_factor = discount_curve.discount_factor(maturity)
        origin_factor = discount_curve.discount_factor(self.env.pricing_date)
        if not all(
            np.isfinite(x) and x > 0 for x in (discount_factor, origin_factor)
        ):
            raise ValueError("discount factors must be finite and positive")
        return payoffs * (discount_factor / origin_factor)

    def _replay_pricer(self) -> MonteCarloPricer:
        """Clone unseeded engines for common draws without mutating callers."""

        if self.process.seed is not None:
            return self
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        return replace(self, process=replace(self.process, seed=seed))

    def _sampling_scheme(self) -> str:
        process = self.process
        if isinstance(process, MultiAssetGeometricBrownianMotion):
            deterministic = bool(np.all(np.asarray(process.volatilities) == 0))
        elif isinstance(process, HestonProcess):
            initial = (
                process.theta
                if self.initial_variance is None
                else self.initial_variance
            )
            deterministic = process.theta == 0 and initial == 0
        else:
            deterministic = process.volatility == 0
            if isinstance(process, JumpDiffusionProcess):
                deterministic = deterministic and (
                    process.jump_intensity == 0
                    or (process.jump_mean == 0 and process.jump_std == 0)
                )
        if deterministic:
            return "deterministic"
        if process.moment_matching:
            return "moment-matched"
        return "antithetic" if process.antithetic else "iid"

    def _result_from_payoffs(
        self, discounted: np.ndarray, deterministic: bool=False
    ) -> MonteCarloResult:
        discounted = np.asarray(discounted, dtype=float)
        if discounted.ndim != 1 or discounted.size != self.paths:
            raise ValueError("payoff must return one value per path")
        if not np.all(np.isfinite(discounted)):
            raise ValueError("discounted payoffs must be finite")
        scheme = "deterministic" if deterministic else self._sampling_scheme()
        return MonteCarloResult(
            price=float(np.mean(discounted)),
            std_error=_sampling_error(discounted, scheme),
            discounted_payoffs=discounted,
            sampling_scheme=scheme,
        )

    @staticmethod
    def _terminal_slice(paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            return paths[:, -1]
        if paths.ndim == 3:
            return paths[:, -1, :]
        raise ValueError("Unexpected paths array shape")

    def price_terminal(
        self,
        payoff: TerminalPayoff,
        maturity: dt.date | dt.datetime,
        spot: float | Sequence[float] | np.ndarray,
    ) -> MonteCarloResult:
        """Simulate terminal payoffs and return PV samples/error metadata."""

        grid, _ = self._prepare_grid(maturity)
        variance = (
            getattr(self.process, "theta", 0.04)
            if self.initial_variance is None
            else self.initial_variance
        )
        if isinstance(self.process, HestonProcess):
            if not isinstance(spot, (int, float)):
                raise TypeError("HestonProcess expects scalar spot")
            spot_scalar = float(spot)
            spot_paths, _ = self.process.simulate_paths(
                spot=spot_scalar,
                variance=variance,
                time_grid=grid,
                paths=self.paths,
            )
            terminal = self._terminal_slice(spot_paths)
            discounted = self._discount(maturity, payoff(terminal))
        elif isinstance(self.process, MultiAssetGeometricBrownianMotion):
            spot_vec = np.asarray(spot, dtype=float)
            spot_paths = self.process.simulate_paths(
                spot=spot_vec,
                time_grid=grid,
                paths=self.paths,
            )
            terminal = self._terminal_slice(spot_paths)
            discounted = self._discount(maturity, payoff(terminal))
        else:
            if not isinstance(spot, (int, float)):
                raise TypeError("GeometricBrownianMotion expects scalar spot")
            spot_scalar = float(spot)
            paths = self.process.simulate_paths(
                spot=spot_scalar,
                time_grid=grid,
                paths=self.paths,
            )
            terminal = self._terminal_slice(paths)
            discounted = self._discount(maturity, payoff(terminal))
        return self._result_from_payoffs(
            discounted, deterministic=grid.size == 1
        )

    def price_path_dependent(
        self,
        payoff: PathPayoff,
        maturity: dt.date | dt.datetime,
        spot: float | Sequence[float] | np.ndarray,
    ) -> MonteCarloResult:
        """Evaluate a path payoff on the chosen finite monitoring grid."""

        grid, _ = self._prepare_grid(maturity)
        variance = (
            getattr(self.process, "theta", 0.04)
            if self.initial_variance is None
            else self.initial_variance
        )
        if isinstance(self.process, HestonProcess):
            if not isinstance(spot, (int, float)):
                raise TypeError("HestonProcess expects scalar spot")
            spot_scalar = float(spot)
            spot_paths, _ = self.process.simulate_paths(
                spot=spot_scalar,
                variance=variance,
                time_grid=grid,
                paths=self.paths,
            )
            discounted = self._discount(maturity, payoff(spot_paths))
        elif isinstance(self.process, MultiAssetGeometricBrownianMotion):
            spot_vec = np.asarray(spot, dtype=float)
            spot_paths = self.process.simulate_paths(
                spot=spot_vec,
                time_grid=grid,
                paths=self.paths,
            )
            discounted = self._discount(maturity, payoff(spot_paths))
        else:
            if not isinstance(spot, (int, float)):
                raise TypeError("GeometricBrownianMotion expects scalar spot")
            spot_scalar = float(spot)
            paths = self.process.simulate_paths(
                spot=spot_scalar, time_grid=grid, paths=self.paths
            )
            discounted = self._discount(maturity, payoff(paths))
        return self._result_from_payoffs(
            discounted, deterministic=grid.size == 1
        )

    def price_european(
        self,
        payoff: TerminalPayoff,
        maturity: dt.date | dt.datetime,
        spot: float,
    ) -> float:
        """Backwards-compatible helper returning only the price."""

        return self.price_terminal(payoff, maturity, spot).price

    def sensitivity(
        self,
        payoff: TerminalPayoff,
        maturity: dt.date | dt.datetime,
        spot: float,
        *,
        bump: float=1e-3,
        order: int=1,
    ) -> float:
        """
        Finite-difference sensitivity (order 1 => delta, order 2 => gamma).
        """

        if order not in (1, 2):
            raise ValueError(
                "Only first and second order sensitivities supported"
            )
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be finite and positive")
        if not np.isfinite(bump) or not 0 < bump < 1:
            raise ValueError("bump must be in (0, 1)")
        pricer = self._replay_pricer()
        up_spot = spot * (1.0 + bump)
        down_spot = spot * (1.0 - bump)
        up = pricer.price_terminal(payoff, maturity, up_spot).price
        down = pricer.price_terminal(payoff, maturity, down_spot).price
        h = spot * bump
        if order == 1:
            return (up - down) / (2 * h)
        if order == 2:
            base = pricer.price_terminal(payoff, maturity, spot).price
            return (up - 2 * base + down) / (h**2)
        raise ValueError("Only first and second order sensitivities supported")
