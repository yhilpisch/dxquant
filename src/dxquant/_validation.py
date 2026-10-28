"""Internal domain validation shared by numerical engines."""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def positive_integer(value: int, name: str) -> None:
    """Reject nonintegral, boolean, or nonpositive sizes."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite_parameters(**values: float) -> None:
    """Reject nonfinite model parameters before numerical operations."""

    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


def simulation_grid(
    values: Sequence[float] | NDArray[np.float64],
    paths: int,
) -> NDArray[np.float64]:
    """Validate a finite increasing grid from zero and positive path count."""

    positive_integer(paths, "paths")
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or not grid.size or grid[0] != 0:
        raise ValueError("time_grid must be 1D and start at 0")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time_grid must be finite")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    return grid
