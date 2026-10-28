# Golden parity tests comparing dxquant outputs against the legacy dx package.
#
# These tests import the frozen ``dx`` reference from the sibling ``dx/`` tree
# and
# require its optional dependencies (see the ``parity`` extra). They are
# skip-safe:
# if ``dx`` cannot be imported, the whole module is skipped so core CI stays
# green.

import datetime as dt
import importlib
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from dxquant.fourier import FourierPricer
from dxquant.processes import HestonProcess, JumpDiffusionProcess
from dxquant.time import uniform_time_grid, year_fractions
from dxquant.valuation.european import (
    black_scholes_call_price,
    black_scholes_put_price,
)

PRICING_DATE = dt.date(2024, 1, 1)
MATURITY_DATE = dt.date(2025, 1, 1)
# Match dx's ACT/365 ttm (366 days because 2024 is a leap year).
TTM = float(year_fractions([PRICING_DATE, MATURITY_DATE])[1])


def _import_dx() -> ModuleType:
    dx_root = Path(__file__).resolve().parents[2] / "dx"
    sys.path.insert(0, str(dx_root))
    if "multiprocess" not in sys.modules:
        stub = ModuleType("multiprocess")
        stub.Pool = lambda *a, **k: None  # type: ignore[attr-defined]
        sys.modules["multiprocess"] = stub
    return importlib.import_module("dx")


try:
    dx = _import_dx()
except Exception as exc:  # pragma: no cover - skip when dx deps missing
    pytest.skip(f"dx import unavailable: {exc}", allow_module_level=True)


# ----------------------------------------------------------------------------
# Black-Scholes-Merton: dxquant analytical helpers vs dx.BSM_european_option
# ----------------------------------------------------------------------------


def _bsm_market_environment(
    dx: ModuleType, spot: float, strike: float, rate: float, vol: float
) -> Any:
    me = dx.market_environment("bs", dt.datetime(2024, 1, 1))
    me.add_constant("initial_value", spot)
    me.add_constant("strike", strike)
    me.add_constant("maturity", dt.datetime(2025, 1, 1))
    me.add_constant("volatility", vol)
    me.add_curve("discount_curve", dx.constant_short_rate("disc", rate))
    return me


@pytest.mark.parametrize("strike", [80.0, 95.0, 100.0, 105.0, 120.0])
def test_bsm_call_parity_vs_dx(strike: float) -> None:
    spot, rate, vol = 100.0, 0.05, 0.2
    me = _bsm_market_environment(dx, spot, strike, rate, vol)
    dx_call = dx.BSM_european_option("call", me).call_value()
    dxa_call = black_scholes_call_price(spot, strike, rate, vol, TTM)
    assert dx_call == pytest.approx(dxa_call, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("strike", [80.0, 95.0, 100.0, 105.0, 120.0])
def test_bsm_put_parity_vs_dx(strike: float) -> None:
    spot, rate, vol = 100.0, 0.05, 0.2
    me = _bsm_market_environment(dx, spot, strike, rate, vol)
    dx_put = dx.BSM_european_option("call", me).put_value()
    dxa_put = black_scholes_put_price(spot, strike, rate, vol, TTM)
    assert dx_put == pytest.approx(dxa_put, rel=1e-10, abs=1e-12)


# ----------------------------------------------------------------------------
# Heston: dxquant Carr-Madan Fourier vs dx H93 Lewis Fourier
# ----------------------------------------------------------------------------


def _heston_dx_price(strike: float) -> float:
    me = dx.market_environment("h93", dt.datetime(2024, 1, 1))
    me.add_constant("initial_value", 100.0)
    me.add_constant("strike", strike)
    me.add_constant("maturity", dt.datetime(2025, 1, 1))
    me.add_constant("kappa", 2.0)
    me.add_constant("theta", 0.04)
    me.add_constant("vol_vol", 0.5)
    me.add_constant("rho", -0.4)
    me.add_constant("volatility", 0.2)  # sqrt(v0); v0 = 0.04
    me.add_curve("discount_curve", dx.constant_short_rate("disc", 0.01))
    return float(dx.H93_call_value(me))


@pytest.mark.parametrize("strike", [80.0, 90.0, 100.0, 110.0, 120.0])
def test_heston_fourier_parity_vs_dx(strike: float) -> None:
    rate, spot, v0 = 0.01, 100.0, 0.04
    heston = HestonProcess(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.4, drift=rate
    )
    phi = heston.characteristic_function(TTM, spot, v0)
    dxa_price = FourierPricer(
        dampening=1.5, integration_limit=200.0
    ).price_call(phi, spot, strike, math.exp(-rate * TTM))
    dx_price = _heston_dx_price(strike)
    # Carr-Madan vs legacy Lewis Fourier: agree within approximately 1%.
    assert dx_price == pytest.approx(dxa_price, rel=1e-2)


# ----------------------------------------------------------------------------
# Merton jump diffusion: dxquant Monte Carlo vs dx M76 analytical Fourier
# ----------------------------------------------------------------------------


def test_jump_diffusion_mc_parity_vs_dx() -> None:
    rate, vol, lamb, mu, delta = 0.02, 0.2, 0.5, -0.5, 0.2
    spot, strike = 100.0, 100.0

    me = dx.market_environment("m76", dt.datetime(2024, 1, 1))
    me.add_constant("initial_value", spot)
    me.add_constant("strike", strike)
    me.add_constant("maturity", dt.datetime(2025, 1, 1))
    me.add_constant("lambda", lamb)
    me.add_constant("mu", mu)
    me.add_constant("delta", delta)
    me.add_constant("volatility", vol)
    me.add_curve("discount_curve", dx.constant_short_rate("disc", rate))
    dx_price = float(dx.M76_call_value(me))

    process = JumpDiffusionProcess(
        drift=rate,
        volatility=vol,
        jump_intensity=lamb,
        jump_mean=mu,
        jump_std=delta,
        seed=42,
    )
    grid = uniform_time_grid(TTM, 64)
    paths = process.simulate_paths(spot, grid, 200000)
    discounted = np.maximum(paths[:, -1] - strike, 0.0) * math.exp(-rate * TTM)
    dxa_price = float(discounted.mean())
    # Compare Monte Carlo with the legacy analytical price; allow 1%.
    assert dx_price == pytest.approx(dxa_price, rel=1e-2)


# ----------------------------------------------------------------------------
# Year fractions: dxquant.year_fractions vs dx.get_year_deltas
# ----------------------------------------------------------------------------


def test_year_fractions_parity_vs_dx() -> None:
    dates = [dt.date(2024, 1, 1), dt.date(2024, 4, 1), dt.date(2025, 1, 1)]
    dx_deltas = dx.get_year_deltas(dates)
    dxa_deltas = year_fractions(dates)
    np.testing.assert_allclose(dxa_deltas, dx_deltas, rtol=1e-12)
