# Migration from DX Analytics

DXQuant is a partial rewrite with different object interfaces and numerical
contracts. It is not a drop-in replacement for the original `dx` library.
Install DXQuant from this repository and use `import dxquant as dx`; the alias
is a local name for DXQuant, not a legacy compatibility layer.

Earlier local versions used `dxa`. Change `from dxa...` imports to
`from dxquant...`, or use the recommended alias. No `dxa` compatibility package
is shipped. The unrelated PyPI distribution named `dxa` is not this library.
`DXAConfig` retains its existing class name within `dxquant`.

| Legacy concept | DXQuant entry point |
|---|---|
| Market environment | `MarketEnvironment` |
| Constant/deterministic rate | `ConstantShortRateCurve`, `InterpolatedZeroCurve` |
| Single/multi-risk processes | `GeometricBrownianMotion`, `HestonProcess`, `JumpDiffusionProcess`, `MultiAssetGeometricBrownianMotion` |
| European valuation | `MonteCarloPricer`, `EuropeanOptionValuation`, analytical call/put helpers |
| American put | `LongstaffSchwartzPricer` |
| Positions/portfolios | `PositionSpec`, `PortfolioPricer` |
| Scenarios/risk | `ScenarioAnalyzer`, `risk_report` |
| Interest-rate products | `InterestRateSwap`, `CIRShortRateProcess`, `cir_zero_coupon` |
| Mean-variance portfolios | `MeanVariancePortfolio` |

Use the [tutorials](notebooks/README.md) for complete examples and read the
[numerical contracts](docs/numerical_contracts.rst) for model-specific domains.
In particular, square-root jumps use additive marks rather than the legacy
level-scaled convention. Portfolio VaR/ES describes discounted-payoff dispersion,
not general future-MTM market risk; joint payoff samples are caller-managed.
Some seeded outputs and uncertainty estimates intentionally differ from legacy
or earlier local implementations. General market bootstrapping, additional
exotics, full legacy parity, and accelerated backends remain future scope.
