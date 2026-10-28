# DXQuant

Quantitative finance in Python: models, valuation, and risk analytics.

DXQuant is an alpha library inspired by DX Analytics. Its current focus is
stochastic models, derivatives valuation, curves, and portfolio analytics.
Broader rates, bonds, and credit products remain future scope. It is a partial
rewrite, not a drop-in replacement for the original `dx` library.

## Installation

Python 3.10 or newer is required. Runtime dependencies are NumPy, pandas,
and SciPy. The current source version is `0.2.0a0`, before its first PyPI release.
Install from GitHub:

```bash
python -m pip install git+https://github.com/yhilpisch/dxquant.git
```

Or install a local clone:

```bash
git clone https://github.com/yhilpisch/dxquant.git
cd dxquant
python -m pip install .
```

The distribution and Python package are both `dxquant`. Recommended usage:

```python
import dxquant as dx

price = dx.black_scholes_call_price(
    spot=100.0, strike=100.0, rate=0.02, vol=0.2, maturity=1.0
)
print(round(price, 4))  # 8.916
```

The unrelated PyPI project named `dxa` is not this library. No `dxa`
compatibility import is provided.

## Tutorials and Google Colab

The [13 tutorials](notebooks/README.md) cover simulation, valuation, calibration,
rates, portfolios, calendars, and numerical validation. Each has an Open in
Colab link and a setup cell that installs missing DXQuant/Matplotlib packages.
Select **Runtime → Run all** on a standard CPU runtime. No GPU, Drive mount,
API credentials, or local datasets are needed. Examples use synthetic inputs.

Before PyPI publication, setup installs GitHub `main`. Local checkouts install
their local package when DXQuant is missing; existing installations are reused.
The maintainer has confirmed a complete Quickstart run on hosted Colab.
The remaining tutorials have local execution checks; full hosted coverage
still needs validation.

Validate all tutorials headlessly from the repository root:

```bash
python tools/validate_colab.py
python tools/validate_colab.py --execute \
    --report /tmp/dxquant-colab-report.json
```

The default performs formal checks without running cells or installing packages.
`--execute` installs the current checkout and runs all tutorials with plotting
in fresh kernels. `--timeout` sets a positive per-cell limit (default 300 seconds).
Failures return a nonzero exit status; source notebooks remain unchanged.
The optional JSON report records versions, revision/dirty status, hashes,
timings, and errors. See the tutorial index for Colab commands.

## Implemented scope

| Area | Implementation |
|---|---|
| Foundation | Calendar dates/day counts, business-day calendars, seeded RNG, interpolation, market environments |
| Curves | Constant/interpolated zero curves, restricted deposit/swap bootstrap, covered projection curves |
| Processes | Single/multi-asset GBM, Heston, Merton jumps, mean-reverting and square-root jump diffusion, SABR, CIR |
| Valuation | Black–Scholes calls/puts, terminal/path Monte Carlo, Asian/barrier/basket payoffs, American put LSM, GBM/Heston Fourier calls |
| Rates | CIR bond pricing/calibration, optional two-curve payer swaps, schedules/day counts/amortizing notionals |
| Portfolio | Quantity/FX aggregation, discounted-payoff VaR/ES, scenarios, risk tables, parallel pricing/Greeks, mean-variance optimization |
| Calibration | Scalar root solving, least-squares fitting, SABR implied-volatility approximation |

Read the [numerical contracts](docs/numerical_contracts.rst) before relying on
prices or risk measures. These document supported domains, finite-grid
approximations, sampling-error availability, projection restrictions, and
caller-managed joint payoff scenarios. Payoff VaR/ES is not future-MTM
market-risk VaR. Full legacy parity, general market/OIS bootstrapping,
additional exotics, live-data adapters, and accelerated backends are not claimed.

The public compatibility surface is the 61 names in `dxquant.__all__`.
`DXAConfig` retains its existing class name. Optional extras are `dev`, `docs`,
`notebooks`, `parity`, and `accelerators`; the latter installs dependencies but
no accelerated backend is implemented. See [migration](MIGRATION.md),
[contributing](CONTRIBUTING.md), and the [changelog](CHANGELOG.md).

## License

AGPL-3.0-or-later. See [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE) for the
complete license and provenance notices.
