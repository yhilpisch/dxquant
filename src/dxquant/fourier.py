# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Fourier-based pricing helpers for European options (Carr-Madan style).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.integrate import quad

from ._validation import positive_integer
from .processes import GeometricBrownianMotion, _DeterministicLogPrice

__all__ = ["CharacteristicFunction", "FourierPricer"]

CharacteristicFunction = Callable[[complex], complex]


def gbm_characteristic_function(
    process: GeometricBrownianMotion,
    maturity: float,
    spot: float,
) -> CharacteristicFunction:
    """
    Closed-form characteristic function for log-price under GBM.
    """

    if (
        not math.isfinite(maturity)
        or maturity < 0
        or not math.isfinite(spot)
        or spot <= 0
    ):
        raise ValueError(
            "positive spot and nonnegative finite maturity required"
        )
    if maturity == 0 or process.volatility == 0:
        return _DeterministicLogPrice(math.log(spot) + process.drift * maturity)
    mu = (process.drift - 0.5 * process.volatility**2) * maturity
    variance = (process.volatility**2) * maturity

    log_spot = math.log(spot)

    def phi(u: complex) -> complex:
        return cmath.exp(1j * u * (log_spot + mu) - 0.5 * variance * u * u)

    return phi


@dataclass
class FourierPricer:
    """
    Implements the Carr-Madan Fourier pricing formula for European calls.

    Integration uses :func:`scipy.integrate.quad` for robust, adaptive accuracy.
    ``steps`` limits quadrature subintervals. Consecutive doubling tail checks
    detect cutoff sensitivity; they are a convergence diagnostic, not a rigorous
    bound on an arbitrary CF tail. The CF must provide the damping moment.
    Exact point masses from the package CF factories bypass oscillatory
    quadrature at expiry/zero volatility. ``spot`` remains a validated
    compatibility argument; the supplied log-price CF contains its level.
    """

    dampening: float = 1.5  # alpha parameter
    integration_limit: float = 150.0
    steps: int = 250
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-8
    max_tail_steps: int = 8

    def __post_init__(self) -> None:
        for value in (
            self.dampening,
            self.integration_limit,
            self.absolute_tolerance,
            self.relative_tolerance,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite Fourier controls required")
        positive_integer(self.steps, "steps")
        positive_integer(self.max_tail_steps, "max_tail_steps")

    def price_call(
        self,
        characteristic_fn: CharacteristicFunction,
        spot: float,
        strike: float,
        discount_factor: float,
    ) -> float:
        """Price a call from a log-price CF with explicit convergence checks."""

        if not all(
            math.isfinite(x) and x > 0 for x in (spot, strike, discount_factor)
        ):
            raise ValueError(
                "spot, strike, and discount factor must be positive finite"
            )
        if isinstance(characteristic_fn, _DeterministicLogPrice):
            terminal = math.exp(characteristic_fn.log_price)
            return discount_factor * max(terminal - strike, 0.0)
        try:
            normalization = characteristic_fn(0)
            mean = characteristic_fn(-1j)
            moment = characteristic_fn(-(self.dampening + 1) * 1j)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError(
                "characteristic-function moment unavailable"
            ) from exc
        if (
            not all(
                math.isfinite(complex(x).real)
                and math.isfinite(complex(x).imag)
                for x in (normalization, mean, moment)
            )
            or abs(normalization - 1) > 1e-9
            or mean.real <= 0
            or abs(mean.imag) > 1e-8 * max(1, abs(mean.real))
            or moment.real <= 0
            or abs(moment.imag) > 1e-8 * max(1, abs(moment.real))
        ):
            raise ValueError(
                "normalized CF and finite real positive moments required"
            )
        alpha = self.dampening
        k = math.log(strike)

        def integrand(u: float) -> float:
            numerator = characteristic_fn(u - (alpha + 1) * 1j)
            denominator = alpha**2 + alpha - u * u + 1j * u * (2 * alpha + 1)
            return (cmath.exp(-1j * u * k) * numerator / denominator).real

        factor = discount_factor * math.exp(-alpha * k) / math.pi

        def integrate(lower: float, upper: float) -> float:
            result = quad(
                integrand,
                lower,
                upper,
                limit=self.steps,
                epsabs=self.absolute_tolerance / max(abs(factor), 1e-300),
                epsrel=self.relative_tolerance,
                full_output=1,
            )
            if len(result) != 3 or not math.isfinite(result[0]):
                raise ValueError("Fourier quadrature failed to converge")
            return float(result[0])

        integral = integrate(0, self.integration_limit)
        left = self.integration_limit
        quiet = 0
        for _ in range(self.max_tail_steps):
            right = left * 2
            tail = integrate(left, right)
            integral += tail
            tolerance = max(
                self.absolute_tolerance,
                self.relative_tolerance * abs(factor * integral),
            )
            quiet = quiet + 1 if abs(factor * tail) <= tolerance else 0
            if quiet >= 2:
                break
            left = right
        else:
            raise ValueError(
                "Fourier tail did not converge; revise integration controls"
            )
        price = factor * integral
        lower = discount_factor * max(mean.real - strike, 0)
        upper = discount_factor * mean.real
        tolerance = max(
            self.absolute_tolerance, self.relative_tolerance * upper
        )
        if price < lower - 10 * tolerance or price > upper + 10 * tolerance:
            raise ValueError(
                "Fourier price violates positive-asset call bounds"
            )
        return float(min(upper, max(lower, price)))
