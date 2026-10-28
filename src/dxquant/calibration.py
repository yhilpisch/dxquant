# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Calibration utilities wrapping SciPy optimizers with DXQuant defaults.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.optimize as sco

__all__ = ["solve_scalar", "LeastSquaresCalibration", "calibrate_least_squares"]


def solve_scalar(
    objective: Callable[[float], float],
    bracket: tuple[float, float],
    tol: float=1e-8,
) -> float:
    """
    Root of a scalar equation using Brent's method.
    """

    if (
        not all(math.isfinite(x) for x in (*bracket, tol))
        or bracket[0] >= bracket[1]
        or tol <= 0
    ):
        raise ValueError(
            "finite ordered bracket and positive tolerance required"
        )
    result = sco.root_scalar(
        objective, bracket=bracket, xtol=tol, method="brentq"
    )
    if not result.converged:
        raise RuntimeError("Scalar solver failed to converge")
    return float(result.root)


@dataclass
class LeastSquaresCalibration:
    """
    Wraps ``scipy.optimize.least_squares`` and stores diagnostics.
    """

    target: np.ndarray
    model: Callable[[np.ndarray], np.ndarray]
    initial_parameters: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    def run(self) -> sco.OptimizeResult:
        """Fit finite residual vectors without implicit shape broadcasting."""

        self.target = np.asarray(self.target, dtype=float)
        self.initial_parameters = np.asarray(
            self.initial_parameters, dtype=float
        )
        for array in (self.target, self.initial_parameters):
            if (
                array.ndim != 1
                or not array.size
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    "target/initial parameters must be finite vectors"
                )
        if self.bounds is not None:
            lower, upper = (np.asarray(b, dtype=float) for b in self.bounds)
            if (
                lower.shape != self.initial_parameters.shape
                or upper.shape != lower.shape
                or np.any(np.isnan(lower))
                or np.any(np.isnan(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError("bounds must align and satisfy lower < upper")
            self.bounds = (lower, upper)

        def residuals(params: np.ndarray) -> np.ndarray:
            fitted = np.asarray(self.model(params), dtype=float)
            if fitted.shape != self.target.shape or not np.all(
                np.isfinite(fitted)
            ):
                raise ValueError("model output must match finite target vector")
            residual = fitted - self.target
            return np.asarray(residual, dtype=float)

        kwargs = {"bounds": self.bounds} if self.bounds is not None else {}
        result = sco.least_squares(residuals, self.initial_parameters, **kwargs)
        if not result.success:
            raise RuntimeError(f"Calibration failed: {result.message}")
        return result


def calibrate_least_squares(
    model: Callable[[np.ndarray], np.ndarray],
    target: Sequence[float] | np.ndarray,
    guess: Sequence[float] | np.ndarray,
    bounds: tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]
    | None=None,
) -> np.ndarray:
    """
    Convenience wrapper returning fitted parameters.
    """

    bounds_array: tuple[np.ndarray, np.ndarray] | None = None
    if bounds is not None:
        lower, upper = bounds
        bounds_array = (
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    lsq = LeastSquaresCalibration(
        target=np.asarray(target, dtype=float),
        model=model,
        initial_parameters=np.asarray(guess, dtype=float),
        bounds=bounds_array,
    )
    result = lsq.run()
    return np.asarray(result.x, dtype=float)
