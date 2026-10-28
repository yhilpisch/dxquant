# Changelog

## 0.2.0a0 — Unreleased alpha

This is the initial DXQuant source release, inspired by DX Analytics. No PyPI
release is claimed. The distribution/import name is `dxquant`; examples use
`import dxquant as dx`.

- Stochastic processes, payoff composition, Monte Carlo valuation, analytical
  Black–Scholes and Fourier benchmarks, calibration, curves, and portfolio tools.
- Correct compound-Poisson sampling and independent jump/diffusion streams;
  supported deterministic/near-zero Heston/CIR limits and SABR equations.
- American inception/expiry exercise; explicit projection/quote-grid domains,
  optional two-curve swaps, calendar dates/day counts, and finite-input checks.
- Caller-certified joint payoff scenarios, quantity/FX-scaled Greeks,
  base-relative scenario PnL, paired antithetic errors, and explicit unavailable
  sampling errors for unsupported single-batch estimators.
- Typed public API, model/domain documentation, 13 self-contained Colab
  tutorials, and headless formal/full notebook validation with optional reports.

These are alpha contracts with finite-grid/model limitations. Corrected seeded
outputs differ from earlier local implementations. Consult the
[numerical contracts](docs/numerical_contracts.rst) and [migration](MIGRATION.md).

## Compatibility policy

The compatibility surface is the 61 names in `dxquant.__all__`, guarded by
`tests/test_public_api.py`. Additional module helpers are not covered merely
because they appear in the API reference. During `0.x`, intentional breaking
public API changes must be documented and increase the minor version.
