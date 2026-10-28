# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Random number generation helpers with deterministic hooks for Monte Carlo
engines.

This module re-implements ``sn_random_numbers`` from the legacy ``dx`` package
while enforcing modern type hints, reproducibility options, and inline
commentary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["sn_random_numbers"]


def sn_random_numbers(
    shape: Sequence[int],
    *,
    antithetic: bool=True,
    moment_matching: bool=True,
    seed: int | None=None,
) -> np.ndarray:
    """
    Generate standard normally distributed pseudo-random numbers.

    Parameters
    ----------
    shape:
        Tuple describing the desired array shape (typically paths, time-steps,
        factors).
    antithetic:
        If ``True`` the function returns antithetic variates along the last axis
        to reduce variance.
    moment_matching:
        If ``True`` the sample mean/variance are adjusted to better match the
        theoretical moments.
    seed:
        Optional random seed used for deterministic Monte Carlo experiments.
    """

    if len(shape) < 1:
        raise ValueError("shape must describe at least one dimension")

    if (
        any(
            isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
            for n in shape
        )
        or shape[-1] < 1
    ):
        raise ValueError(
            "shape must contain nonnegative integer sizes and paths"
        )
    rng = np.random.default_rng(
        seed
    )  # use Generator API for better statistical properties

    if antithetic:
        if len(shape) < 2:
            raise ValueError("antithetic generation expects at least two axes")

        half_columns = int(
            np.ceil(shape[-1] / 2)
        )  # determine how many samples are needed
        base_shape = (*shape[:-1], half_columns)
        base_draws = rng.standard_normal(size=base_shape)
        mirrored = -base_draws  # create antithetic counterpart
        samples = np.concatenate((base_draws, mirrored), axis=-1)
        samples = samples[..., : shape[-1]]  # trim in case of odd counts
    else:
        samples = rng.standard_normal(size=shape)

    if moment_matching and shape[-1] > 1:
        samples = samples - samples.mean(
            axis=-1, keepdims=True
        )  # enforce zero mean
        std = samples.std(axis=-1, keepdims=True)
        std[std == 0.0] = 1.0  # guard against degenerate cases
        samples = samples / std  # scale to unit variance

    return np.asarray(samples)
