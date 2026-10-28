# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
European option valuation helpers using Monte Carlo or analytical pricing.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol

from dxquant.curves import ConstantShortRateCurve
from dxquant.monte_carlo import MonteCarloPricer
from dxquant.payoffs import EuropeanCallPayoff
from dxquant.processes import GeometricBrownianMotion, HestonProcess
from dxquant.time import year_fraction

__all__ = [
    "black_scholes_call_price",
    "black_scholes_put_price",
    "EuropeanOptionValuation",
]


class AnalyticalPricer(Protocol):
    """Callable interface for an analytical strike-price benchmark."""

    def __call__(self, strike: float) -> float:
        """Return the analytical value at a strike."""
        ...


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
) -> float:
    """
    Black-Scholes-Merton European call price (continuous compounding).

    Edge cases are handled explicitly so the function stays well-defined for
    ``maturity == 0`` and ``vol == 0`` (deterministic terminal spot):

    * ``maturity == 0`` -> intrinsic value ``max(spot - strike, 0)``.
    * ``vol == 0``      -> ``max(spot - strike * exp(-r * t), 0)``.
    """

    if not all(math.isfinite(x) for x in (spot, strike, rate, vol, maturity)):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if vol < 0:
        raise ValueError("vol must be non-negative")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")

    if maturity == 0.0:
        return max(spot - strike, 0.0)  # expired -> intrinsic
    if vol == 0.0:
        # Risk-neutral terminal spot is deterministic: S_T = spot * exp(r * t).
        discount = math.exp(-rate * maturity)
        return max(spot - strike * discount, 0.0)

    sigma_sqrt_t = vol * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * vol**2) * maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return spot * _norm_cdf(d1) - strike * math.exp(
        -rate * maturity
    ) * _norm_cdf(d2)


def black_scholes_put_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
) -> float:
    """
    Black-Scholes-Merton European put price (continuous compounding).

    Same edge-case handling as :func:`black_scholes_call_price`:
    ``maturity == 0`` -> ``max(strike - spot, 0)``; ``vol == 0`` ->
    ``max(strike * exp(-r * t) - spot, 0)``.
    """

    if not all(math.isfinite(x) for x in (spot, strike, rate, vol, maturity)):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if vol < 0:
        raise ValueError("vol must be non-negative")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")

    if maturity == 0.0:
        return max(strike - spot, 0.0)
    if vol == 0.0:
        discount = math.exp(-rate * maturity)
        return max(strike * discount - spot, 0.0)

    call = black_scholes_call_price(spot, strike, rate, vol, maturity)
    return call - spot + strike * math.exp(-rate * maturity)  # put-call parity


# Backwards-compatible alias for code that imported the previous private helper.
_black_scholes_call_price = black_scholes_call_price


@dataclass
class EuropeanOptionValuation:
    """
    Thin wrapper around Monte Carlo pricer with optional analytical benchmark.
    """

    pricer: MonteCarloPricer
    payoff: EuropeanCallPayoff
    maturity: dt.date
    spot: float

    def _maturity_years(self) -> float:
        return year_fraction(
            self.pricer.env.pricing_date, self.maturity, self.pricer.day_count
        )

    def price_mc(self) -> float:
        """Return the Monte Carlo terminal call PV."""

        return self.pricer.price_terminal(
            self.payoff, maturity=self.maturity, spot=self.spot
        ).price

    def delta(self) -> float:
        """Return replayed central-difference spot sensitivity."""

        return self.pricer.sensitivity(
            self.payoff, self.maturity, self.spot, order=1
        )

    def gamma(self) -> float:
        """Return replayed second-order spot sensitivity."""

        return self.pricer.sensitivity(
            self.payoff, self.maturity, self.spot, order=2
        )

    def analytical_price(self) -> float | None:
        """Return the GBM/constant-curve lognormal benchmark when supported."""

        process = self.pricer.process
        maturity_years = self._maturity_years()
        discount = self.pricer.env.get_curve(self.pricer.discount_curve_key)
        if isinstance(process, GeometricBrownianMotion) and isinstance(
            discount, ConstantShortRateCurve
        ):
            price = black_scholes_call_price(
                self.spot,
                self.payoff.strike,
                process.drift,
                process.volatility,
                maturity_years,
            )
            df = discount.discount_factor(
                self.maturity
            ) / discount.discount_factor(self.pricer.env.pricing_date)
            return price * math.exp(process.drift * maturity_years) * df
        if isinstance(process, HestonProcess):
            return None  # use Fourier module externally
        return None
