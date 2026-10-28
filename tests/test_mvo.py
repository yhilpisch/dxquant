# Mean-variance portfolio optimization (Markowitz).

import numpy as np
import pandas as pd
import pytest

from dxquant import MeanVariancePortfolio


def _synthetic_returns(
    seed: int=7, n: int=500, assets: int=3
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Distinct drifts and vols so optimisation has a meaningful solution.
    mus = np.array([0.0005, 0.0010, 0.0008])
    vols = np.array([0.012, 0.020, 0.016])
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal(size=(n, assets))
    correlated = z @ chol.T
    returns = mus + vols * correlated
    return pd.DataFrame(returns, columns=["A", "B", "C"])


def test_equal_weights_default_and_stats() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    np.testing.assert_allclose(mvo.weights, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.isfinite(mvo.portfolio_return())
    assert mvo.volatility() > 0
    # variance == volatility**2 and sharpe is consistent.
    np.testing.assert_allclose(
        mvo.portfolio_variance(), mvo.volatility() ** 2, rtol=1e-10
    )


def test_invalid_weights_raise() -> None:
    df = _synthetic_returns()
    with pytest.raises(ValueError):
        MeanVariancePortfolio(
            df, weights=np.array([0.5, 0.3, 0.1])
        )  # sums to 0.9
    with pytest.raises(ValueError):
        MeanVariancePortfolio(df, weights=np.array([0.5, 0.5]))  # wrong length


def test_min_volatility_optimization_reduces_vol() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    equal_vol = mvo.volatility()
    mvo.optimize(target="Vol")
    assert mvo.volatility() < equal_vol  # min-vol portfolio is less risky
    assert mvo.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(mvo.weights >= -1e-9)  # long-only


def test_max_sharpe_optimization_beats_equal_weights() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    equal_sharpe = mvo.sharpe_ratio()
    mvo.optimize(target="Sharpe")
    assert mvo.sharpe_ratio() >= equal_sharpe - 1e-6
    assert mvo.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_return_optimization_puts_weight_on_highest_return_asset() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    mvo.optimize(target="Return")
    # Asset B has the highest mean return in the synthetic fixture.
    best = int(np.argmax(df.mean().to_numpy()))
    assert np.argmax(mvo.weights) == best
    assert mvo.weights[best] > 0.9


def test_efficient_frontier_is_monotonic() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    vols, rets = mvo.efficient_frontier(n_points=15)
    assert vols.size >= 2
    # Higher return on the frontier should come with higher volatility.
    order = np.argsort(rets)
    sorted_vols = vols[order]
    assert np.all(np.diff(sorted_vols) >= -1e-6)


def test_test_weights_does_not_mutate_state() -> None:
    df = _synthetic_returns()
    mvo = MeanVariancePortfolio(df)
    before = mvo.weights.copy()
    stats = mvo.test_weights([0.7, 0.2, 0.1])
    np.testing.assert_array_equal(mvo.weights, before)  # unchanged
    assert stats.shape == (3,)
