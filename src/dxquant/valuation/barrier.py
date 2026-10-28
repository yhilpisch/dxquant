# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Barrier option valuation helpers (path-dependent Monte Carlo).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from dxquant.monte_carlo import MonteCarloPricer
from dxquant.payoffs import PathPayoff

__all__ = ["BarrierOptionValuation"]


@dataclass
class BarrierOptionValuation:
    """Finite-grid path-payoff valuation wrapper."""

    pricer: MonteCarloPricer
    payoff: PathPayoff
    maturity: dt.date | dt.datetime
    spot: float

    def price(self) -> float:
        """Return the discounted Monte Carlo path-payoff mean."""

        return self.pricer.price_path_dependent(
            self.payoff,
            maturity=self.maturity,
            spot=self.spot,
        ).price
