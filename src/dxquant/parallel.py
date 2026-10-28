# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Parallel valuation helpers for independent positions.

This is the counterpart to ``dx.valuation.parallel_valuation``. It uses
:mod:`concurrent.futures` process pools, with a
sequential execution when ``max_workers <= 1``. The default uses CPU workers.
Each position carries its own pricer, so positions are fully independent tasks.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

from .portfolio import PortfolioPricer, PositionSpec, _fx_rate, _validate_ids

__all__ = ["parallel_value", "parallel_greeks"]


def _price_one(
    spec: PositionSpec, base_currency: str, fx_rates: dict[str, float] | None
) -> tuple[str, float]:
    pricer = PortfolioPricer()
    result = pricer.price(
        [spec], base_currency=base_currency, fx_rates=fx_rates
    )
    return spec.position_id, result.position_results[0].price


def _greek_one(
    spec: PositionSpec,
    order: int,
    bump: float,
    base_currency: str,
    fx_rates: dict[str, float] | None,
) -> tuple[str, float]:
    # Only terminal, single-asset positions support bump sensitivities.
    pricer = spec.pricer
    spot = spec.spot
    if not isinstance(spot, (int, float)) or spec.path_dependent:
        return spec.position_id, float("nan")
    greek = float(
        pricer.sensitivity(
            spec.payoff, spec.maturity, float(spot), bump=bump, order=order
        )
    )
    return spec.position_id, greek * spec.quantity * _fx_rate(
        spec, base_currency, fx_rates
    )


def parallel_value(
    specs: Sequence[PositionSpec],
    *,
    max_workers: int | None=None,
    base_currency: str="BASE",
    fx_rates: dict[str, float] | None=None,
) -> dict[str, float]:
    """
    Price independent positions in parallel; returns ``{position_id: price}``.
    """

    _validate_ids(specs)
    if max_workers is not None and (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    if not specs:
        return {}
    if max_workers is None:
        max_workers = max(1, min(len(specs), os.cpu_count() or 1))
    if max_workers <= 1:
        return dict(_price_one(spec, base_currency, fx_rates) for spec in specs)

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(
            _price_one,
            list(specs),
            [base_currency] * len(specs),
            [fx_rates] * len(specs),
        )
        return dict(results)


def parallel_greeks(
    specs: Sequence[PositionSpec],
    *,
    order: int=1,
    bump: float=1e-3,
    max_workers: int | None=None,
    base_currency: str="BASE",
    fx_rates: dict[str, float] | None=None,
) -> dict[str, float]:
    """
    Compute finite-difference sensitivities per position in parallel.

    ``order=1`` returns delta, ``order=2`` returns gamma. Only single-asset,
    non-path-dependent positions are supported (others return ``nan``).
    Results include position quantity/FX in base currency. Position IDs
    must be unique. Unsupported multi-asset/path-dependent Greeks are NaN.
    """

    if order not in (1, 2) or not math.isfinite(bump) or not 0 < bump < 1:
        raise ValueError("order must be 1/2 and bump must be in (0, 1)")
    _validate_ids(specs)
    if max_workers is not None and (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise ValueError("max_workers must be a positive integer")
    if not specs:
        return {}
    if max_workers is None:
        max_workers = max(1, min(len(specs), os.cpu_count() or 1))
    if max_workers <= 1:
        return dict(
            _greek_one(spec, order, bump, base_currency, fx_rates)
            for spec in specs
        )

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(
            _greek_one,
            list(specs),
            [order] * len(specs),
            [bump] * len(specs),
            [base_currency] * len(specs),
            [fx_rates] * len(specs),
        )
        return dict(results)
