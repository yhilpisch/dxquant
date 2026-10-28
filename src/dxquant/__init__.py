# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# Original package: https://github.com/yhilpisch/dx
# AI-Powered by different LLMs

"""
DXQuant — modern derivatives analytics package rebuilt from the legacy ``dx``
reference.

Numerical modules expose explicit reproducibility controls.  Every public
function or class should
carry docstrings plus inline comments to make the numerical intent auditable.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from .bootstrapping import DepositQuote, SwapQuote, bootstrap_zero_curve
from .calendar import BusinessDayCalendar
from .calibration import (
    LeastSquaresCalibration,
    calibrate_least_squares,
    solve_scalar,
)
from .curves import ConstantShortRateCurve, InterpolatedZeroCurve
from .env import MarketEnvironment
from .fourier import FourierPricer, gbm_characteristic_function
from .interpolate import Spline1D, linear_interpolate
from .monte_carlo import MonteCarloPricer, MonteCarloResult
from .mvo import MeanVariancePortfolio
from .parallel import parallel_greeks, parallel_value
from .payoffs import (
    ArithmeticAsianCallPayoff,
    BasketCallPayoff,
    EuropeanCallPayoff,
    PathPayoff,
    TerminalPayoff,
    UpAndOutBarrierCallPayoff,
)
from .portfolio import (
    PortfolioPricer,
    PortfolioResult,
    PositionSpec,
    Scenario,
    ScenarioAnalyzer,
    scale_spot,
)
from .processes import (
    GeometricBrownianMotion,
    HestonProcess,
    JumpDiffusionProcess,
    MeanRevertingDiffusion,
    MultiAssetGeometricBrownianMotion,
    SABRProcess,
    SquareRootJumpDiffusion,
    sabr_implied_volatility,
)
from .random import sn_random_numbers
from .rates.cir import CIRShortRateProcess, cir_zero_coupon
from .rates.forward import ForwardCurve, bootstrap_forward_curve
from .rates.swap import InterestRateSwap
from .risk import PortfolioRiskReport, risk_report
from .time import (
    DayCount,
    ensure_datetime_array,
    uniform_time_grid,
    year_fraction,
    year_fractions,
)
from .valuation import (
    BarrierOptionValuation,
    EuropeanOptionValuation,
    LongstaffSchwartzPricer,
    MultiAssetEuropeanValuation,
    black_scholes_call_price,
    black_scholes_put_price,
)

__all__ = [
    "__version__",
    "DXAConfig",
    "build_default_config",
    "ensure_datetime_array",
    "year_fractions",
    "year_fraction",
    "DayCount",
    "sn_random_numbers",
    "DepositQuote",
    "SwapQuote",
    "bootstrap_zero_curve",
    "cir_zero_coupon",
    "CIRShortRateProcess",
    "InterestRateSwap",
    "ForwardCurve",
    "bootstrap_forward_curve",
    "solve_scalar",
    "LeastSquaresCalibration",
    "calibrate_least_squares",
    "BusinessDayCalendar",
    "Spline1D",
    "linear_interpolate",
    "ConstantShortRateCurve",
    "InterpolatedZeroCurve",
    "MarketEnvironment",
    "GeometricBrownianMotion",
    "MultiAssetGeometricBrownianMotion",
    "HestonProcess",
    "JumpDiffusionProcess",
    "MeanRevertingDiffusion",
    "SquareRootJumpDiffusion",
    "SABRProcess",
    "sabr_implied_volatility",
    "MonteCarloPricer",
    "MonteCarloResult",
    "PortfolioPricer",
    "PortfolioResult",
    "PositionSpec",
    "Scenario",
    "ScenarioAnalyzer",
    "scale_spot",
    "parallel_value",
    "parallel_greeks",
    "MeanVariancePortfolio",
    "PortfolioRiskReport",
    "risk_report",
    "FourierPricer",
    "gbm_characteristic_function",
    "TerminalPayoff",
    "PathPayoff",
    "EuropeanCallPayoff",
    "ArithmeticAsianCallPayoff",
    "UpAndOutBarrierCallPayoff",
    "BasketCallPayoff",
    "uniform_time_grid",
    "EuropeanOptionValuation",
    "LongstaffSchwartzPricer",
    "BarrierOptionValuation",
    "MultiAssetEuropeanValuation",
    "black_scholes_call_price",
    "black_scholes_put_price",
]

# Set the canonical package version so tooling/tests can assert compatibility
# quickly.
__version__ = "0.2.0a0"


@dataclass(frozen=True)
class DXAConfig:
    """
    Frozen configuration fields; nested metadata objects remain caller-managed.

    Parameters
    ----------
    rng_seed:
        Optional random seed to enforce reproducibility. ``None`` delegates
        seeding to callers.
    metadata:
        Free-form mapping for downstream hooks (e.g., curve names, pricing
        contexts).
    """

    rng_seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self) -> MutableMapping[str, Any]:
        """
        Convert the config into a mutable mapping for consumers that expect
        dict-like inputs.
        """
        payload: MutableMapping[str, Any] = {"metadata": dict(self.metadata)}
        if self.rng_seed is not None:
            payload["rng_seed"] = (
                self.rng_seed
            )  # keep deterministic Monte Carlo across services
        return payload


def build_default_config() -> DXAConfig:
    """
    Helper that returns a baseline config object and centralizes future
    defaults.

    Having a dedicated constructor makes it trivial to inject new fields while
    keeping existing
    call sites untouched, and acts as a single inspection point for auditors.
    """

    # Default to ``None`` so that each engine can decide whether to seed locally
    # or accept a seed.
    return DXAConfig(rng_seed=None, metadata={})
