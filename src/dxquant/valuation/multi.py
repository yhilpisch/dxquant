# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Multi-asset valuation helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dxquant.curves import ConstantShortRateCurve
from dxquant.processes import MultiAssetGeometricBrownianMotion
from dxquant.time import uniform_time_grid

__all__ = ["MultiAssetEuropeanValuation"]


PayoffFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MultiAssetEuropeanValuation:
    """
    Monte Carlo valuation for European-style payoffs depending on multiple
    assets.
    """

    process: MultiAssetGeometricBrownianMotion
    discount_curve: ConstantShortRateCurve
    payoff: PayoffFn
    maturity_years: float
    spot: Sequence[float]
    paths: int = 50000
    steps: int = 64

    def price(self) -> float:
        """
        Value a terminal multi-asset payoff with constant-rate discounting.
        """

        grid = uniform_time_grid(self.maturity_years, self.steps)
        simulated = self.process.simulate_paths(
            spot=self.spot,
            time_grid=grid,
            paths=self.paths,
        )
        terminal = simulated[:, -1, :]
        payoffs = self.payoff(terminal)
        discount = math.exp(-self.discount_curve.rate * self.maturity_years)
        return float(np.mean(payoffs) * discount)
