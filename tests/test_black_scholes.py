# Analytical Black-Scholes helpers and edge-case hardening.

import math

import pytest

from dxquant import black_scholes_call_price, black_scholes_put_price
from dxquant.valuation.european import (
    black_scholes_call_price as bs_call_via_valuation,
)


def test_call_price_matches_textbook_formula() -> None:
    spot, strike, rate, vol, t = 100.0, 100.0, 0.05, 0.2, 1.0
    # Manual computation via the standard BS formula.
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * t) / (
        vol * math.sqrt(t)
    )
    d2 = d1 - vol * math.sqrt(t)

    def n(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    expected = spot * n(d1) - strike * math.exp(-rate * t) * n(d2)
    assert black_scholes_call_price(
        spot, strike, rate, vol, t
    ) == pytest.approx(expected)


def test_put_call_parity_holds() -> None:
    spot, strike, rate, vol, t = 105.0, 100.0, 0.03, 0.25, 2.0
    call = black_scholes_call_price(spot, strike, rate, vol, t)
    put = black_scholes_put_price(spot, strike, rate, vol, t)
    assert call - put == pytest.approx(spot - strike * math.exp(-rate * t))


def test_maturity_zero_returns_intrinsic() -> None:
    assert black_scholes_call_price(105.0, 100.0, 0.02, 0.2, 0.0) == 5.0
    assert black_scholes_call_price(95.0, 100.0, 0.02, 0.2, 0.0) == 0.0
    assert black_scholes_put_price(105.0, 100.0, 0.02, 0.2, 0.0) == 0.0
    assert black_scholes_put_price(95.0, 100.0, 0.02, 0.2, 0.0) == 5.0


def test_zero_vol_returns_deterministic_discounted_payoff() -> None:
    spot, strike, rate, t = 105.0, 100.0, 0.02, 1.5
    # Risk-neutral S_T = spot * exp(r t); discounted payoff = max(spot - K
    # e^{-rt}, 0).
    assert black_scholes_call_price(
        spot, strike, rate, 0.0, t
    ) == pytest.approx(max(spot - strike * math.exp(-rate * t), 0.0))
    assert black_scholes_put_price(spot, strike, rate, 0.0, t) == pytest.approx(
        max(strike * math.exp(-rate * t) - spot, 0.0)
    )


def test_zero_vol_out_of_the_money_is_zero() -> None:
    assert black_scholes_call_price(90.0, 100.0, 0.02, 0.0, 1.0) == 0.0
    assert black_scholes_put_price(110.0, 100.0, 0.02, 0.0, 1.0) == 0.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        black_scholes_call_price(-1.0, 100.0, 0.02, 0.2, 1.0)
    with pytest.raises(ValueError):
        black_scholes_call_price(100.0, 0.0, 0.02, 0.2, 1.0)
    with pytest.raises(ValueError):
        black_scholes_call_price(100.0, 100.0, 0.02, -0.1, 1.0)
    with pytest.raises(ValueError):
        black_scholes_call_price(100.0, 100.0, 0.02, 0.2, -1.0)


def test_public_helper_reachable_from_top_level_and_valuation() -> None:
    # Both the top-level re-export and the valuation-module path resolve to the
    # same function.
    assert black_scholes_call_price is bs_call_via_valuation
