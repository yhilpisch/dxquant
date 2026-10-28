# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
American option valuation using Longstaff-Schwartz regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dxquant._validation import positive_integer
from dxquant.curves import ConstantShortRateCurve
from dxquant.processes import GeometricBrownianMotion
from dxquant.time import uniform_time_grid

__all__ = ["LongstaffSchwartzPricer"]


@dataclass
class LongstaffSchwartzPricer:
    """
    Least-squares Monte Carlo pricer for single-asset American options.
    """

    process: GeometricBrownianMotion
    discount_curve: ConstantShortRateCurve
    paths: int = 50000
    steps: int = 50

    def __post_init__(self) -> None:
        positive_integer(self.paths, "paths")
        positive_integer(self.steps, "steps")

    def price_american_put(
        self, spot: float, strike: float, maturity_years: float
    ) -> float:
        """Price a put with exercise at inception and every grid date."""

        if not all(math.isfinite(x) for x in (spot, strike, maturity_years)):
            raise ValueError("spot, strike, and maturity must be finite")
        if spot <= 0 or strike <= 0 or maturity_years < 0:
            raise ValueError(
                "positive spot/strike and nonnegative maturity required"
            )
        if maturity_years == 0:
            return max(strike - spot, 0.0)
        grid = uniform_time_grid(maturity_years, self.steps)
        simulated = self.process.simulate_paths(
            spot=spot, time_grid=grid, paths=self.paths
        )
        dt = maturity_years / self.steps
        discount = math.exp(-self.discount_curve.rate * dt)
        cashflows = np.maximum(strike - simulated[:, -1], 0.0)

        for step in range(self.steps - 1, 0, -1):
            cashflows *= discount
            intrinsic = np.maximum(strike - simulated[:, step], 0.0)
            itm = intrinsic > 1e-12
            if not np.any(itm):
                continue
            x_values = simulated[itm, step]
            y_values = cashflows[itm]
            design = np.vstack(
                [np.ones_like(x_values), x_values, x_values**2]
            ).T
            coeffs, _, _, _ = np.linalg.lstsq(design, y_values, rcond=None)
            continuation = (
                coeffs[0] + coeffs[1] * x_values + coeffs[2] * x_values**2
            )
            exercise = intrinsic[itm] > continuation
            cashflows[itm] = np.where(exercise, intrinsic[itm], cashflows[itm])

        continuation_at_zero = float(np.mean(cashflows) * discount)
        return max(strike - spot, continuation_at_zero, 0.0)
