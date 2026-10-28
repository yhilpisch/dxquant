# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Portfolio risk reporting on top of :class:`dxquant.portfolio.PortfolioPricer`.

This module re-implements the spirit of
``dx.valuation.derivatives_portfolio.risk_report``
as a deterministic, typed function that returns a :class:`pandas.DataFrame` of
per-position
and portfolio-level risk metrics (price, delta, gamma, VaR, ES).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import PortfolioPricer, PositionSpec, _replay_specs

__all__ = ["PortfolioRiskReport", "risk_report"]


@dataclass
class PortfolioRiskReport:
    """
    Compute a per-position risk table for a portfolio of Monte Carlo positions.

    Delta and gamma are estimated by bumping each position's spot and re-pricing
    that single position (finite differences). VaR/ES come from the position's
    discounted-payoff distribution.
    """

    pricer: PortfolioPricer
    specs: Sequence[PositionSpec]
    bump: float = 1e-2
    var_confidence: float = 0.95

    def _single_price(
        self,
        spec: PositionSpec,
        base_currency: str,
        fx_rates: dict[str, float] | None,
    ) -> float:
        return self.pricer.price(
            [spec], base_currency=base_currency, fx_rates=fx_rates
        ).price

    def _position_greeks(
        self,
        spec: PositionSpec,
        base_currency: str,
        fx_rates: dict[str, float] | None,
    ) -> tuple[float, float, float]:
        spot = spec.spot
        if isinstance(spot, (list, tuple, np.ndarray)):
            # Multi-asset spots: greeks via scalar bump not well-defined here.
            base = self._single_price(spec, base_currency, fx_rates)
            return base, float("nan"), float("nan")
        if not isinstance(spot, (int, float)):
            raise TypeError("spot must be scalar to compute greeks")
        spot_scalar = float(spot)
        h = spot_scalar * self.bump
        if h == 0.0:
            raise ValueError("spot must be non-zero to compute greeks")
        from dataclasses import replace

        up = replace(spec, spot=spot_scalar * (1.0 + self.bump))
        down = replace(spec, spot=spot_scalar * (1.0 - self.bump))
        base = self._single_price(spec, base_currency, fx_rates)
        p_up = self._single_price(up, base_currency, fx_rates)
        p_down = self._single_price(down, base_currency, fx_rates)
        delta = (p_up - p_down) / (2.0 * h)  # central difference
        gamma = (p_up - 2.0 * base + p_down) / (h * h)  # second difference
        return base, delta, gamma

    def compute(
        self,
        *,
        base_currency: str="BASE",
        fx_rates: dict[str, float] | None=None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per position plus a portfolio total row.
        """

        if not math.isfinite(self.bump) or not 0 < self.bump < 1:
            raise ValueError("bump must be in (0, 1)")
        if (
            not math.isfinite(self.var_confidence)
            or not 0 < self.var_confidence < 1
        ):
            raise ValueError("confidence must be in (0, 1)")
        specs = _replay_specs(self.specs)
        aggregate = self.pricer.price(
            specs, base_currency=base_currency, fx_rates=fx_rates
        )
        rows: list[dict[str, float | str]] = []
        for spec in specs:
            pos_result = next(
                r
                for r in aggregate.position_results
                if r.spec.position_id == spec.position_id
            )
            base, delta, gamma = self._position_greeks(
                spec, base_currency, fx_rates
            )
            # Per-position VaR/ES from the (scaled, FX-converted) payoff vector.
            losses = pos_result.price - pos_result.discounted_payoffs
            var = float(np.percentile(losses, self.var_confidence * 100.0))
            tail = losses[losses >= var]
            es = float(np.mean(tail)) if tail.size else var
            rows.append(
                {
                    "position_id": spec.position_id,
                    "quantity": spec.quantity,
                    "currency": base_currency,
                    "position_currency": spec.currency,
                    "price": pos_result.price,
                    "std_error": pos_result.std_error,
                    "delta": delta,
                    "gamma": gamma,
                    f"var_{self.var_confidence * 100:g}": var,
                    "es": es,
                }
            )

        rows.append(
            {
                "position_id": "PORTFOLIO",
                "quantity": "",
                "currency": base_currency,
                "price": aggregate.price,
                "std_error": aggregate.std_error,
                "delta": float("nan"),
                "gamma": float("nan"),
                f"var_{self.var_confidence * 100:g}": aggregate.value_at_risk(
                    self.var_confidence
                )
                if aggregate.risk_scenarios_aligned
                else float("nan"),
                "es": (
                    aggregate.expected_shortfall(self.var_confidence)
                    if aggregate.risk_scenarios_aligned
                    else float("nan")
                ),
            }
        )
        table = pd.DataFrame(rows)
        table.attrs["risk_basis"] = (
            "discounted-payoff dispersion, not horizon MTM PnL"
        )
        table.attrs["joint_scenarios"] = aggregate.risk_scenarios_aligned
        return table


def risk_report(
    pricer: PortfolioPricer,
    specs: Sequence[PositionSpec],
    *,
    bump: float=1e-2,
    var_confidence: float=0.95,
    base_currency: str="BASE",
    fx_rates: dict[str, float] | None=None,
) -> pd.DataFrame:
    """Convenience wrapper returning the risk table for a portfolio."""

    report = PortfolioRiskReport(
        pricer=pricer, specs=specs, bump=bump, var_confidence=var_confidence
    )
    return report.compute(base_currency=base_currency, fx_rates=fx_rates)
