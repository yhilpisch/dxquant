# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
Portfolio aggregation utilities built on top of Monte Carlo results.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from .calendar import _to_date
from .monte_carlo import MonteCarloPricer, MonteCarloResult, _sampling_error
from .payoffs import PathPayoff, TerminalPayoff

__all__ = [
    "PositionSpec",
    "PositionResult",
    "PortfolioResult",
    "PortfolioPricer",
]


@dataclass(frozen=True)
class PositionSpec:
    """
    Defines one quantity/currency-scaled position.

    scenario_set identifies caller-certified ordered joint samples. It does
    not construct a correlation model or a future mark-to-market risk horizon.
    Multi-position payoff risk also requires a common maturity/pricing date.
    """

    position_id: str
    pricer: Any
    payoff: TerminalPayoff | PathPayoff
    maturity: dt.date | dt.datetime
    spot: float | Sequence[float] | np.ndarray
    quantity: float = 1.0
    path_dependent: bool = False
    currency: str = "BASE"
    scenario_set: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position_id, str) or not self.position_id:
            raise ValueError("position_id must be a nonempty string")
        if self.position_id == "PORTFOLIO":
            raise ValueError("PORTFOLIO is reserved for aggregate reports")
        if not math.isfinite(self.quantity):
            raise ValueError("quantity must be finite")
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency must be nonempty")
        if self.scenario_set is not None and not self.scenario_set:
            raise ValueError("scenario_set must be nonempty when supplied")


@dataclass
class PositionResult:
    """
    Stores pricing output for a single position (after quantity scaling).
    """

    spec: PositionSpec
    price: float
    std_error: float
    discounted_payoffs: np.ndarray

    @property
    def quantity(self) -> float:
        """Return the quantity from the immutable position specification."""

        return self.spec.quantity


@dataclass
class PortfolioResult:
    """
    Aggregate PV and discounted-payoff dispersion, not horizon MTM VaR.

    Risk percentiles require aligned scenarios. With no joint declaration,
    price remains a sum of position PVs but aggregate risk/error is unavailable.
    Raw payoff vectors must not be treated as certified joint samples unless
    risk_scenarios_aligned is true.
    """

    price: float
    std_error: float
    discounted_payoffs: np.ndarray
    position_results: list[PositionResult] = field(default_factory=list)
    base_currency: str = "BASE"
    risk_scenarios_aligned: bool = True
    sampling_scheme: str = "unknown"

    def _require_aligned(self) -> None:
        if not self.risk_scenarios_aligned:
            raise ValueError("joint scenario_set and common horizon required")

    def percentile(self, q: float) -> float:
        """Percentile of the aligned discounted-payoff distribution."""

        self._require_aligned()
        if not math.isfinite(q) or not 0 <= q <= 100:
            raise ValueError("percentile must be in [0, 100]")
        return float(np.percentile(self.discounted_payoffs, q))

    def value_at_risk(self, confidence: float=0.95) -> float:
        """Loss quantile at the requested confidence."""

        self._require_aligned()
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence must be in (0, 1)")
        losses = self.price - self.discounted_payoffs
        return float(np.percentile(losses, confidence * 100.0))

    def expected_shortfall(self, confidence: float=0.95) -> float:
        """Average loss beyond the VaR threshold."""

        losses = self.price - self.discounted_payoffs
        var_threshold = self.value_at_risk(confidence)
        tail = losses[losses >= var_threshold]
        if tail.size == 0:
            return var_threshold
        tail_mean = float(np.mean(tail))
        return max(var_threshold, tail_mean)


def _validate_ids(specs: Sequence[PositionSpec]) -> None:
    ids = [spec.position_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("position IDs must be unique")


def _fx_rate(
    spec: PositionSpec, base_currency: str, fx_rates: dict[str, float] | None
) -> float:
    if spec.currency == base_currency:
        return 1.0
    if not fx_rates or spec.currency not in fx_rates:
        raise ValueError(f"Missing FX rate for currency {spec.currency}")
    fx = fx_rates[spec.currency]
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError("FX rates must be finite and positive")
    return fx


def _replay_specs(specs: Sequence[PositionSpec]) -> list[PositionSpec]:
    cache: dict[int, MonteCarloPricer] = {}
    result: list[PositionSpec] = []
    for spec in specs:
        if isinstance(spec.pricer, MonteCarloPricer):
            key = id(spec.pricer)
            if key not in cache:
                cache[key] = spec.pricer._replay_pricer()
            spec = replace(spec, pricer=cache[key])
        result.append(spec)
    return result


class PortfolioPricer:
    """Batch prices multiple positions and aggregates portfolio risk metrics."""

    def price(
        self,
        specs: Sequence[PositionSpec],
        *,
        base_currency: str="BASE",
        fx_rates: dict[str, float] | None=None,
    ) -> PortfolioResult:
        """
        Aggregate quantity/FX-scaled PVs and certify payoff-risk availability.
        """

        if not specs:
            raise ValueError("specs must contain at least one position")

        _validate_ids(specs)
        aggregated: np.ndarray | None = None
        position_results: list[PositionResult] = []
        sample_size: int | None = None
        schemes: list[str] = []

        for spec in specs:
            result = self._price_position(spec)
            fx = _fx_rate(spec, base_currency, fx_rates)
            vector = np.asarray(result.discounted_payoffs, dtype=float)
            if (
                vector.ndim != 1
                or not vector.size
                or not np.all(np.isfinite(vector))
            ):
                raise ValueError(
                    "position payoff samples must be nonempty finite vectors"
                )
            schemes.append(result.sampling_scheme)
            scaled = result.discounted_payoffs * spec.quantity * fx

            if aggregated is None:
                aggregated = scaled.copy()
                sample_size = aggregated.size
            else:
                if scaled.size != sample_size:
                    raise ValueError(
                        "All positions must share identical path counts"
                    )
                aggregated += scaled

            position_results.append(
                PositionResult(
                    spec=spec,
                    price=result.price * spec.quantity * fx,
                    std_error=(
                        0.0 if spec.quantity == 0
                        else result.std_error * abs(spec.quantity) * fx
                    ),
                    discounted_payoffs=scaled,
                )
            )

        assert aggregated is not None  # defensive for type checkers
        price = float(np.mean(aggregated))
        maturities = {
            s.maturity.date()
            if isinstance(s.maturity, dt.datetime)
            else s.maturity
            for s in specs
        }
        origins = {
            getattr(getattr(s.pricer, "env", None), "pricing_date", None)
            for s in specs
        }
        origins = {_to_date(date) if date is not None else None
                   for date in origins}
        labels = {s.scenario_set for s in specs}
        deterministic = all(
            r.std_error == 0 and np.all(r.discounted_payoffs == r.price)
            for r in position_results
        )
        aligned = len(specs) == 1 or (
            len(maturities) == 1
            and len(origins) == 1
            and (deterministic or (len(labels) == 1 and None not in labels))
        )
        if deterministic:
            scheme = "deterministic"
        elif "moment-matched" in schemes:
            scheme = "moment-matched"
        elif "unknown" in schemes:
            scheme = "unknown"
        elif "antithetic" in schemes:
            scheme = "antithetic"
        else:
            scheme = "iid"
        std_error = (
            _sampling_error(aggregated, scheme) if aligned else float("nan")
        )

        return PortfolioResult(
            price=price,
            std_error=std_error,
            discounted_payoffs=aggregated,
            position_results=position_results,
            base_currency=base_currency,
            risk_scenarios_aligned=aligned,
            sampling_scheme=scheme,
        )

    @staticmethod
    def _price_position(spec: PositionSpec) -> MonteCarloResult:
        if spec.path_dependent:
            path_payoff = cast(PathPayoff, spec.payoff)
            result = spec.pricer.price_path_dependent(
                path_payoff, maturity=spec.maturity, spot=spec.spot
            )
            return cast(MonteCarloResult, result)
        terminal_payoff = cast(TerminalPayoff, spec.payoff)
        result = spec.pricer.price_terminal(
            terminal_payoff, maturity=spec.maturity, spot=spec.spot
        )
        return cast(MonteCarloResult, result)


# Scenario helpers -----------------------------------------------------------

ScenarioFn = Callable[[PositionSpec], PositionSpec]


@dataclass(frozen=True)
class Scenario:
    """Named deterministic modifier of position specifications."""

    name: str
    modifier: ScenarioFn


@dataclass
class ScenarioResult:
    """Scenario portfolio value and its unshocked base value."""

    name: str
    result: PortfolioResult
    base_price: float | None = None

    def pnl(self) -> float:
        """Scenario value less the unshocked base portfolio value."""

        if self.base_price is None:
            raise ValueError("base_price required for scenario PnL")
        return self.result.price - self.base_price


class ScenarioAnalyzer:
    """Reprice a base portfolio under named position shocks."""

    def __init__(
        self, pricer: PortfolioPricer, specs: Sequence[PositionSpec]
    ) -> None:
        self.pricer = pricer
        self.specs = list(specs)

    def run(
        self,
        scenarios: Sequence[Scenario],
        *,
        base_currency: str="BASE",
        fx_rates: dict[str, float] | None=None,
    ) -> list[ScenarioResult]:
        """Return scenario values/PnL using replayed supported MC engines."""

        specs = _replay_specs(self.specs)
        base = self.pricer.price(
            specs, base_currency=base_currency, fx_rates=fx_rates
        ).price
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            shocked = [scenario.modifier(spec) for spec in specs]
            results.append(
                ScenarioResult(
                    scenario.name,
                    self.pricer.price(
                        shocked, base_currency=base_currency, fx_rates=fx_rates
                    ),
                    base,
                )
            )
        return results


def scale_spot(position_id: str, factor: float) -> ScenarioFn:
    """Create a positive multiplicative spot shock for one position."""

    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("spot scale must be finite and positive")

    def modifier(spec: PositionSpec) -> PositionSpec:
        if spec.position_id != position_id:
            return spec
        new_spot: float | Sequence[float] | np.ndarray
        if isinstance(spec.spot, (list, tuple)):
            new_spot = [s * factor for s in spec.spot]
        elif isinstance(spec.spot, np.ndarray):
            new_spot = spec.spot * factor
        else:
            new_spot = cast(float, spec.spot) * factor
        return replace(spec, spot=new_spot)

    return modifier
