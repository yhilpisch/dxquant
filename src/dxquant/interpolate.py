# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Interpolation helpers inspired by the ``dx`` spline utilities.

The original implementation scattered spline construction across valuation
classes. DXQuant centralizes the logic so all modules can share the same,
well-documented abstraction with deterministic behavior.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate as sci
from numpy.typing import NDArray

from ._validation import positive_integer

FloatArray = NDArray[np.float64]

__all__ = ["Spline1D", "linear_interpolate"]


def _as_float_array(values: Sequence[float] | FloatArray) -> FloatArray:
    """
    Convert user input into a 1D float array and validate dimensionality.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Input array must be one-dimensional")
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("Input array must be nonempty and finite")
    return array


@dataclass(slots=True)
class Spline1D:
    """
    Cubic (or lower-order) spline interpolator with derivative support.

    Parameters
    ----------
    x:
        Strictly increasing grid of x-coordinates.
    y:
        Function values on the ``x`` grid.
    degree:
        Desired spline order (default 3 for cubic).  Automatically reduced when
        ``len(x)`` is small.
    extrapolate:
        When ``False`` the spline raises ``ValueError`` for out-of-domain
        evaluations.
    """

    x: FloatArray = field(init=False)
    y: FloatArray = field(init=False)
    degree: int = 3
    extrapolate: bool = True
    _tck: tuple[np.ndarray, np.ndarray, int] = field(init=False, repr=False)

    def __init__(
        self,
        x: Sequence[float] | FloatArray,
        y: Sequence[float] | FloatArray,
        *,
        degree: int=3,
        extrapolate: bool=True,
    ) -> None:
        x_arr = _as_float_array(x)
        y_arr = _as_float_array(y)

        if x_arr.size != y_arr.size:
            raise ValueError("x and y arrays must share the same length")
        if np.any(np.diff(x_arr) <= 0):
            raise ValueError("x grid must be strictly increasing")
        positive_integer(degree, "degree")
        if degree > 5 or x_arr.size < 2:
            raise ValueError("spline requires two nodes and degree in [1, 5]")

        actual_degree = min(
            degree, x_arr.size - 1
        )  # reduce order if sample too small
        tck = sci.splrep(x_arr, y_arr, k=actual_degree, s=0)

        object.__setattr__(self, "x", x_arr)
        object.__setattr__(self, "y", y_arr)
        object.__setattr__(self, "degree", actual_degree)
        object.__setattr__(self, "extrapolate", extrapolate)
        object.__setattr__(self, "_tck", tck)

    def _evaluate(self, points: Iterable[float], derivative: int) -> FloatArray:
        values_in = points if isinstance(points, np.ndarray) else list(points)
        points_arr = np.atleast_1d(np.asarray(values_in, dtype=float))
        if points_arr.ndim != 1 or not np.all(np.isfinite(points_arr)):
            raise ValueError("query points must be a finite vector")
        if (
            isinstance(derivative, bool)
            or not isinstance(derivative, (int, np.integer))
            or not 0 <= derivative <= self.degree
        ):
            raise ValueError("derivative order outside spline degree")
        ext = 0 if self.extrapolate else 2  # 2 => ValueError on out-of-bounds
        values = sci.splev(points_arr, self._tck, der=derivative, ext=ext)
        return np.asarray(values, dtype=float)

    def value(self, points: Iterable[float]) -> FloatArray:
        """
        Evaluate the spline at ``points``.
        """

        return self._evaluate(points, derivative=0)

    def derivative(self, points: Iterable[float], order: int=1) -> FloatArray:
        """
        Evaluate the ``order``-th derivative (default first derivative).
        """

        return self._evaluate(points, derivative=order)


def linear_interpolate(
    x: Sequence[float] | FloatArray,
    y: Sequence[float] | FloatArray,
    points: Iterable[float],
) -> FloatArray:
    """
    Lightweight linear interpolator mirroring ``np.interp`` semantics with
    validation.
    """

    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y arrays must share the same length")
    if np.any(np.diff(x_arr) <= 0):
        raise ValueError("x grid must be strictly increasing")

    values_in = points if isinstance(points, np.ndarray) else list(points)
    query = np.atleast_1d(np.asarray(values_in, dtype=float))
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError("query points must be a finite vector")
    results = np.interp(query, x_arr, y_arr, left=y_arr[0], right=y_arr[-1])
    return np.asarray(results, dtype=float)
