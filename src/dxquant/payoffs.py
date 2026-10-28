# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Instrument payoff abstractions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]

__all__ = [
    "TerminalPayoff",
    "PathPayoff",
    "EuropeanCallPayoff",
    "ArithmeticAsianCallPayoff",
    "UpAndOutBarrierCallPayoff",
    "BasketCallPayoff",
]


def _payoff_array(values: FloatArray, dimension: int) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != dimension
        or not array.size
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            "payoff inputs must be nonempty finite arrays of the right shape"
        )
    return array


class TerminalPayoff(Protocol):
    """Callable interface mapping terminal values to payoff vectors."""

    def __call__(self, terminal_values: FloatArray) -> FloatArray:
        """Map terminal values to one payoff per simulation path."""
        ...


class PathPayoff(Protocol):
    """Callable interface consuming full simulated paths (path x time)."""

    def __call__(self, path_values: FloatArray) -> FloatArray:
        """Map path observations to one payoff per simulation path."""
        ...


@dataclass(frozen=True)
class EuropeanCallPayoff:
    """Simple max(S-K, 0) payoff."""

    strike: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strike):
            raise ValueError("strike must be finite")

    def __call__(self, terminal_values: FloatArray) -> FloatArray:
        """Return positive terminal intrinsic values."""

        values = _payoff_array(terminal_values, 1)
        return np.asarray(np.maximum(values - self.strike, 0.0), dtype=float)


@dataclass(frozen=True)
class ArithmeticAsianCallPayoff:
    """Payoff based on the arithmetic average of the full price path."""

    strike: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.strike):
            raise ValueError("strike must be finite")

    def __call__(self, path_values: FloatArray) -> FloatArray:
        """Average all grid observations, including inception, then exercise."""

        path_values = _payoff_array(path_values, 2)
        averages = np.mean(path_values, axis=1)
        return np.asarray(np.maximum(averages - self.strike, 0.0), dtype=float)


@dataclass(frozen=True)
class UpAndOutBarrierCallPayoff:
    """Knock-out call option with an upper barrier."""

    strike: float
    barrier: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.strike, self.barrier)):
            raise ValueError("strike and barrier must be finite")

    def __call__(self, path_values: FloatArray) -> FloatArray:
        """
        Exercise terminal calls unless a monitored value reaches the barrier.
        """

        path_values = _payoff_array(path_values, 2)
        knocked_out = np.max(path_values, axis=1) >= self.barrier
        terminal = path_values[:, -1]
        intrinsic = np.maximum(terminal - self.strike, 0.0)
        intrinsic[knocked_out] = 0.0
        return np.asarray(intrinsic, dtype=float)


@dataclass(frozen=True)
class BasketCallPayoff:
    """Call on weighted sum of terminal asset prices."""

    weights: FloatArray
    strike: float

    def __init__(self, weights: Sequence[float], strike: float) -> None:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if not weights_arr.size or not np.all(np.isfinite(weights_arr)):
            raise ValueError("weights must be nonempty and finite")
        if not math.isfinite(strike):
            raise ValueError("strike must be finite")
        weights_arr = weights_arr.copy()
        weights_arr.setflags(write=False)
        object.__setattr__(self, "weights", weights_arr)
        object.__setattr__(self, "strike", strike)

    def __call__(self, terminal_values: FloatArray) -> FloatArray:
        """Apply weights to terminal assets before exercising the call."""

        if terminal_values.ndim == 1:
            raise ValueError(
                "basket payoff expects multi-asset terminal values"
            )
        terminal_values = _payoff_array(terminal_values, 2)
        if terminal_values.shape[1] != self.weights.size:
            raise ValueError("weights must match the asset dimension")
        basket = terminal_values @ self.weights
        return np.asarray(np.maximum(basket - self.strike, 0.0), dtype=float)
