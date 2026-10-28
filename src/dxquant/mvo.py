# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Mean-variance portfolio optimization (Markowitz).

This is the counterpart to ``dx.portfolio.mean_variance_portfolio``. Unlike
the legacy version it does not fetch data from a URL; callers supply a returns
DataFrame so the module stays deterministic and offline-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.optimize as sco

from ._validation import positive_integer

__all__ = ["MeanVariancePortfolio"]


@dataclass
class MeanVariancePortfolio:
    """
    Markowitz mean-variance portfolio over the assets in ``returns``.

    Parameters
    ----------
    returns:
        DataFrame of asset returns (typically log returns), one column per
        asset.
    weights:
        Initial portfolio weights (must sum to 1). Defaults to equal weights.
    periods_per_year:
        Annualisation factor for returns/covariance (252 for daily data).
    risk_free_rate:
        Risk-free rate used in Sharpe-ratio optimisation (default 0).
    """

    returns: pd.DataFrame
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))
    periods_per_year: float = 252.0
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("returns must be a DataFrame")
        if self.returns.shape[0] < 2 or self.returns.shape[1] < 1:
            raise ValueError("returns require at least two rows and one asset")
        if not self.returns.columns.is_unique:
            raise ValueError("asset labels must be unique")
        data = self.returns.to_numpy(dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError("returns must be finite and complete")
        self.returns = self.returns.copy()
        if not math.isfinite(self.risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.weights = np.asarray(self.weights, dtype=float).copy()
        self._mean = self.returns.mean().to_numpy()  # per-period mean returns
        self._cov = self.returns.cov().to_numpy()  # per-period covariance
        self._n_assets = self.returns.shape[1]
        if self.weights.size == 0:
            self.weights = np.full(self._n_assets, 1.0 / self._n_assets)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (self._n_assets,) or not np.all(
            np.isfinite(self.weights)
        ):
            raise ValueError("weights length must match number of assets")
        if not math.isclose(float(self.weights.sum()), 1.0, abs_tol=1e-6):
            raise ValueError("weights must sum to 1")
        if (
            not math.isfinite(self.periods_per_year)
            or self.periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be positive")

    # -- statistics ----------------------------------------------------------

    def _stats(self, weights: np.ndarray) -> np.ndarray:
        """Return ``[annual_return, annual_vol, sharpe]`` for given weights."""

        w = np.asarray(weights, dtype=float)
        annual_return = float(np.sum(self._mean * w) * self.periods_per_year)
        annual_var = float(w @ (self._cov * self.periods_per_year) @ w)
        annual_vol = math.sqrt(max(annual_var, 0.0))
        sharpe = (
            (annual_return - self.risk_free_rate) / annual_vol
            if annual_vol > 0
            else 0.0
        )
        return np.array([annual_return, annual_vol, sharpe])

    def portfolio_return(self) -> float:
        """Return annualized mean return for the current weights."""

        return float(self._stats(self.weights)[0])

    def portfolio_variance(self) -> float:
        """Return annualized covariance-based portfolio variance."""

        return float(self._stats(self.weights)[1] ** 2)

    def volatility(self) -> float:
        """Return annualized portfolio standard deviation."""

        return float(self._stats(self.weights)[1])

    def sharpe_ratio(self) -> float:
        """Return annualized excess return divided by volatility."""

        return float(self._stats(self.weights)[2])

    def test_weights(self, weights: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Statistics for a candidate weight vector without updating the portfolio.
        """

        w = self._checked_weights(weights)
        return self._stats(w)

    def set_weights(self, weights: Sequence[float] | np.ndarray) -> None:
        """Validate and copy a finite fully invested weight vector."""

        self.weights = self._checked_weights(weights)

    def _checked_weights(
        self, weights: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        w = np.asarray(weights, dtype=float).copy()
        if w.shape != (self._n_assets,) or not np.all(np.isfinite(w)):
            raise ValueError("weights length must match number of assets")
        if not math.isclose(float(w.sum()), 1.0, abs_tol=1e-6):
            raise ValueError("weights must sum to 1")
        return w

    # -- optimisation --------------------------------------------------------

    def optimize(
        self,
        target: str="Sharpe",
        constraint: float | None=None,
        constraint_type: str="Exact",
    ) -> np.ndarray:
        """
        Optimise weights for ``target`` in ``{"Sharpe", "Vol", "Return"}``.

        For ``"Vol"``/``"Return"`` an optional ``constraint`` on the other
        statistic is supported (``constraint_type`` = ``"Exact"`` or
        ``"Bound"``).
        Returns the optimal weight vector (and updates ``self.weights``).
        """

        weights = self._optimal_weights(target, constraint, constraint_type)
        self.weights = weights
        return weights

    def _optimal_weights(
        self, target: str, constraint: float | None, constraint_type: str
    ) -> np.ndarray:
        if constraint_type not in ("Exact", "Bound"):
            raise ValueError("constraint_type must be Exact or Bound")
        if constraint is not None and not math.isfinite(constraint):
            raise ValueError("constraint must be finite")
        if target == "Sharpe" and constraint is not None:
            raise ValueError("Sharpe target does not accept a constraint")
        cons: list[dict[str, object]] = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        ]

        def obj_sharpe(w: np.ndarray) -> float:
            return -float(self._stats(w)[2])  # maximise Sharpe

        def obj_vol(w: np.ndarray) -> float:
            return float(self._stats(w)[1])  # minimise volatility

        def obj_return(w: np.ndarray) -> float:
            return -float(self._stats(w)[0])  # maximise return

        if target == "Sharpe":
            obj = obj_sharpe
        elif target == "Vol":
            obj = obj_vol
            if constraint is not None:
                cons.append(
                    self._constraint("ret", constraint, constraint_type)
                )
        elif target == "Return":
            obj = obj_return
            if constraint is not None:
                cons.append(
                    self._constraint("vol", constraint, constraint_type)
                )
        else:
            raise ValueError(f"Unknown target {target!r}")

        bounds = tuple((0.0, 1.0) for _ in range(self._n_assets))
        start = np.full(self._n_assets, 1.0 / self._n_assets)
        result = sco.minimize(
            obj, start, method="SLSQP", bounds=bounds, constraints=cons
        )
        if not result.success:
            raise RuntimeError(f"Optimisation failed: {result.message}")
        return np.asarray(np.clip(result.x, 0.0, 1.0), dtype=float)

    def _constraint(
        self, stat: str, value: float, kind: str
    ) -> dict[str, object]:
        idx = 0 if stat == "ret" else 1
        if kind == "Exact":
            return {
                "type": "eq",
                "fun": lambda x, i=idx, v=value: self._stats(x)[i] - v,
            }
        if kind == "Bound":
            # For Vol target: return >= value (ineq). For Return target: vol <=
            # value.
            if stat == "ret":
                return {
                    "type": "ineq",
                    "fun": lambda x, v=value: self._stats(x)[0] - v,
                }
            return {
                "type": "ineq",
                "fun": lambda x, v=value: v - self._stats(x)[1],
            }
        raise ValueError(
            f"constraint_type must be 'Exact' or 'Bound', not {kind!r}"
        )

    def efficient_frontier(
        self, n_points: int=50
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(volatilities, returns)`` along the efficient frontier."""

        positive_integer(n_points, "n_points")
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        min_vol_w = self._optimal_weights("Vol", None, "Exact")
        min_vol = self._stats(min_vol_w)[1]
        max_ret_w = self._optimal_weights("Return", None, "Exact")
        max_ret = self._stats(max_ret_w)[0]
        min_ret = self._stats(
            self._optimal_weights("Return", min_vol, "Exact")
        )[0]

        lo, hi = min(min_ret, max_ret), max(min_ret, max_ret)
        targets = np.linspace(lo, hi, n_points)
        vols, rets = [], []
        for t in targets:
            try:
                w = self._optimal_weights("Vol", float(t), "Exact")
            except RuntimeError:
                continue
            stats = self._stats(w)
            rets.append(float(stats[0]))
            vols.append(float(stats[1]))
        return np.asarray(vols, dtype=float), np.asarray(rets, dtype=float)
