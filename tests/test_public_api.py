# Public API surface guard.

import importlib.metadata

import dxquant as dx


def test_all_is_list_of_strings_without_duplicates() -> None:
    assert isinstance(dx.__all__, list)
    assert all(isinstance(name, str) for name in dx.__all__)
    assert len(dx.__all__) == len(set(dx.__all__))  # no duplicates


def test_every_exported_name_is_resolvable() -> None:
    missing = [name for name in dx.__all__ if not hasattr(dx, name)]
    assert missing == [], f"names in __all__ but not importable: {missing}"


def test_version_matches_package_metadata() -> None:
    assert "__version__" in dx.__all__
    assert dx.__version__ == importlib.metadata.version("dxquant")


# Representative public names that must stay importable from the top level.
REQUIRED_EXPORTS = {
    # core
    "MarketEnvironment",
    "ConstantShortRateCurve",
    "InterpolatedZeroCurve",
    "year_fractions",
    "year_fraction",
    "DayCount",
    "sn_random_numbers",
    # processes
    "GeometricBrownianMotion",
    "HestonProcess",
    "JumpDiffusionProcess",
    "MeanRevertingDiffusion",
    "SquareRootJumpDiffusion",
    "SABRProcess",
    "sabr_implied_volatility",
    # valuation / analytics
    "MonteCarloPricer",
    "FourierPricer",
    "gbm_characteristic_function",
    "EuropeanOptionValuation",
    "LongstaffSchwartzPricer",
    "black_scholes_call_price",
    "black_scholes_put_price",
    # portfolio / risk
    "PortfolioPricer",
    "PortfolioResult",
    "PositionSpec",
    "risk_report",
    "parallel_value",
    "parallel_greeks",
    "MeanVariancePortfolio",
    # rates
    "InterestRateSwap",
    "ForwardCurve",
    "bootstrap_forward_curve",
    "bootstrap_zero_curve",
    "cir_zero_coupon",
    "CIRShortRateProcess",
}


def test_required_exports_present() -> None:
    missing = sorted(REQUIRED_EXPORTS - set(dx.__all__))
    assert missing == [], (
        f"required exports missing from dx.__all__: {missing}"
    )


def test_public_api_size_guard() -> None:
    # Guards against accidental additions/removals. When the public API changes
    # intentionally, update this count alongside dx.__all__.
    assert len(dx.__all__) == 61
