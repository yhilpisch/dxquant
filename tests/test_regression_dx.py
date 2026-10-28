# Regression checks against legacy dx package when available.

import datetime as dt
import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from dxquant.valuation.european import black_scholes_call_price


def _import_dx() -> ModuleType:
    """
    Attempt to import the legacy `dx` package from the sibling directory.
    If optional dependencies like multiprocess are missing, install a
    lightweight
    stub so import succeeds; otherwise skip regression.
    """

    dx_root = Path(__file__).resolve().parents[2] / "dx"
    sys.path.insert(0, str(dx_root))

    if "multiprocess" not in sys.modules:
        stub = ModuleType("multiprocess")

        def _unavailable(*args: object, **kwargs: object) -> None:
            raise RuntimeError("multiprocess not available in test harness")

        stub.Pool = _unavailable  # type: ignore[attr-defined]
        sys.modules["multiprocess"] = stub

    return importlib.import_module("dx")


def test_gbm_european_pricer_matches_dx_analytic() -> None:
    try:
        dx = _import_dx()
    except Exception as exc:  # pragma: no cover - skip when dx deps missing
        pytest.skip(f"dx import unavailable: {exc}")

    pricing_date = np.datetime64("2024-01-01")
    maturity = np.datetime64("2025-01-01")
    strike = 100.0
    spot = 105.0
    rate = 0.02
    vol = 0.25

    me = dx.market_environment("bs", pricing_date.item())
    me.add_constant("initial_value", spot)
    me.add_constant("strike", strike)
    me.add_constant("maturity", maturity.item())
    me.add_constant("volatility", vol)
    me.add_curve("discount_curve", dx.constant_short_rate("disc", rate))

    dx_price = dx.BSM_european_option("call", me).call_value()

    maturity_date: dt.date = maturity.astype(object)
    pricing_date_date: dt.date = pricing_date.astype(object)
    ttm = (
        dt.date(maturity_date.year, maturity_date.month, maturity_date.day)
        - dt.date(
            pricing_date_date.year,
            pricing_date_date.month,
            pricing_date_date.day,
        )
    ).days / 365.0
    dxa_price = black_scholes_call_price(
        spot=spot, strike=strike, rate=rate, vol=vol, maturity=ttm
    )

    assert dx_price == pytest.approx(dxa_price, rel=1e-9, abs=1e-12)
