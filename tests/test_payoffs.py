import numpy as np

from dxquant.payoffs import (
    ArithmeticAsianCallPayoff,
    BasketCallPayoff,
    EuropeanCallPayoff,
    UpAndOutBarrierCallPayoff,
)


def test_european_call_payoff() -> None:
    payoff = EuropeanCallPayoff(strike=100.0)
    out = payoff(np.array([110.0, 90.0]))
    np.testing.assert_array_equal(out, np.array([10.0, 0.0]))


def test_arithmetic_asian_call_payoff() -> None:
    payoff = ArithmeticAsianCallPayoff(strike=100.0)
    paths = np.array([[100.0, 105.0, 110.0], [80.0, 90.0, 95.0]])
    np.testing.assert_array_equal(payoff(paths), np.array([5.0, 0.0]))


def test_up_and_out_barrier_call_payoff() -> None:
    payoff = UpAndOutBarrierCallPayoff(strike=100.0, barrier=130.0)
    paths = np.array(
        [
            [100.0, 110.0, 120.0],  # below barrier -> active
            [100.0, 135.0, 140.0],  # barrier breached -> zero payoff
        ]
    )
    np.testing.assert_array_equal(payoff(paths), np.array([20.0, 0.0]))


def test_basket_call_payoff() -> None:
    payoff = BasketCallPayoff(weights=[0.6, 0.4], strike=100.0)
    terminal = np.array([[120.0, 80.0], [80.0, 70.0]])
    basket_vals = terminal @ np.array([0.6, 0.4])
    expected = np.maximum(basket_vals - 100.0, 0.0)
    np.testing.assert_array_equal(payoff(terminal), expected)
